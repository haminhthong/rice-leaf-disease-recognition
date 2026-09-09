from pathlib import Path
from unittest.mock import patch

import pytest

from rice_leaf_detection.pipeline import PipelineOrchestrator, PipelineStatus


def test_pipeline_stages_definition() -> None:
    """Kiểm tra PipelineOrchestrator khai báo đủ stage canonical."""
    orchestrator = PipelineOrchestrator()
    stages = orchestrator.get_stages()

    expected_stages = {
        "data",
        "train",
        "evaluate",
        "tune_policy",
        "errors",
        "test",
        "export",
    }
    assert set(stages.keys()) == expected_stages

    # Stage test phải yêu cầu confirmation
    assert stages["test"].requires_confirmation is True
    assert stages["data"].requires_confirmation is False


def test_pipeline_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    """Kiểm tra hàm dry_run in ra kế hoạch thực thi đầy đủ mà không gặp lỗi."""
    orchestrator = PipelineOrchestrator()
    orchestrator.dry_run()

    captured = capsys.readouterr().out
    assert "PIPELINE EXECUTION PLAN (DRY-RUN)" in captured
    assert "[Stage: DATA]" in captured
    assert "[Stage: TRAIN]" in captured
    assert "[Stage: EXPORT]" in captured


def test_pipeline_invalid_stage() -> None:
    """Kiểm tra ngoại lệ ValueError khi chọn stage không tồn tại."""
    orchestrator = PipelineOrchestrator()
    with pytest.raises(ValueError, match="Stage không hợp lệ"):
        orchestrator.execute_stage("unknown_stage")


def test_pipeline_missing_inputs_blocks_execution(tmp_path: Path) -> None:
    """Kiểm tra pipeline trả về trạng thái FAILED khi thiếu tệp đầu vào tiên quyết."""
    orchestrator = PipelineOrchestrator(
        config_path=Path("configs/default.yaml"),
        output_dir=tmp_path / "non_existent_data",
        runs_dir=tmp_path / "non_existent_runs",
    )

    # Stage train yêu cầu data.yaml phải tồn tại
    status: PipelineStatus = orchestrator.execute_stage("train")
    assert status.status == "FAILED"
    assert "thiếu inputs tiên quyết" in status.message


def test_pipeline_test_stage_requires_confirmation() -> None:
    """Kiểm tra stage test bị khóa nếu không có cờ xác nhận."""
    orchestrator = PipelineOrchestrator()
    # Mock inputs để stage test không bị fail ở bước kiểm tra file
    with patch.object(Path, "exists", return_value=True):
        status = orchestrator.execute_stage("test", confirm_final_test=False)
        assert status.status == "FAILED"
        assert "Tập Test bị khóa" in status.message


def test_pipeline_execute_stage_success() -> None:
    """Kiểm tra thực thi stage thành công khi runner chạy mượt mà."""
    orchestrator = PipelineOrchestrator()

    mock_output_path = Path("artifacts/best.pt")
    with patch.object(Path, "exists", return_value=True):
        with patch.object(orchestrator, "_run_data_stage", return_value=[mock_output_path]):
            status = orchestrator.execute_stage("data")
            assert status.status == "SUCCESS"
            assert status.produced_artifacts == [mock_output_path]
            assert status.duration_sec >= 0.0


def test_pipeline_run_all_success() -> None:
    """Kiểm tra run_all gọi tuần tự các stage chuẩn."""
    orchestrator = PipelineOrchestrator()

    with patch.object(
        orchestrator,
        "execute_stage",
        return_value=PipelineStatus(
            stage_name="mock",
            status="SUCCESS",
            duration_sec=0.1,
            message="OK",
        ),
    ) as mock_exec:
        results = orchestrator.run_all(confirm_final_test=True)
        assert len(results) == 7
        assert all(r.status == "SUCCESS" for r in results)
        assert mock_exec.call_count == 7
