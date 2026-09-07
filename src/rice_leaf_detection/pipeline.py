"""Pipeline Orchestrator cho Dự án Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition).

Module này đóng vai trò là bộ điều phối trung tâm (Central Pipeline Controller),
kết nối và tự động hóa toàn bộ vòng đời ML theo chuẩn MLOps:
1. data: Chuẩn hóa nhãn, lọc trùng SHA-256, gom nhóm pHash (BK-Tree), Group-aware Stratified Split.
2. train: Huấn luyện mô hình YOLOv8 với siêu tham số quản lý từ file cấu hình YAML.
3. evaluate: Đánh giá hiệu năng trên Validation set, tính mAP50, mAP50-95 và per-class metrics.
4. errors: Phân tích phân bố lỗi theo kích thước tổn thương và bảng phân loại Error Taxonomy.
5. test: Đánh giá mô hình cuối cùng trên tập Test bị khóa (--confirm-final-test).
6. export: Xuất model + policy + metadata sang artifact versioned và kiểm định parity.

Stage ``compare`` vẫn tồn tại để đọc các run cũ, nhưng không nằm trong canonical
``run_all`` vì orchestrator không được tự chọn một file ``best.pt`` ngẫu nhiên.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_config
from .constants import CLASS_NAMES, DEFAULT_ARCHIVES, SEED
from .utils import configure_utf8_console, seed_everything, sha256_file, write_json

logger = logging.getLogger("rice_leaf_pipeline")


@dataclass
class StageDefinition:
    """Định nghĩa một công đoạn (Stage) trong Pipeline."""

    name: str
    description: str
    inputs: list[Path]
    outputs: list[Path]
    runner: Callable[..., Any]
    requires_confirmation: bool = False


@dataclass
class PipelineStatus:
    """Trạng thái thực thi của một Stage."""

    stage_name: str
    status: str  # "PENDING", "RUNNING", "SUCCESS", "SKIPPED", "FAILED"
    duration_sec: float = 0.0
    message: str = ""
    produced_artifacts: list[Path] = field(default_factory=list)


class PipelineOrchestrator:
    """Bộ điều phối Pipeline chính kiểm soát DAG dependencies và thực thi các stage."""

    def __init__(
        self,
        config_path: Path = Path("configs/default.yaml"),
        output_dir: Path = Path("data/processed/rice_leaf_detection"),
        runs_dir: Path = Path("runs"),
        artifacts_dir: Path = Path("artifacts"),
    ) -> None:
        self.config_path = config_path
        self.output_dir = output_dir
        self.runs_dir = runs_dir
        self.artifacts_dir = artifacts_dir
        self.config = load_config(config_path) if config_path.exists() else None

    def get_stages(self) -> dict[str, StageDefinition]:
        """Khai báo danh sách các stage và quan hệ phụ thuộc dữ liệu (Data Lineage)."""
        data_yaml = self.output_dir / "data.yaml"
        manifest_csv = self.output_dir / "manifest.csv"
        audit_json = self.output_dir / "audit_report.json"

        # Đường dẫn mặc định cho training weights
        architecture = self.config.model.architecture if self.config else "yolov8s"
        train_run_dir = self.runs_dir / "train" / f"{architecture}_640"
        best_weights = train_run_dir / "weights" / "best.pt"
        experiments_csv = self.runs_dir / "evaluate" / "experiments.csv"
        error_summary = self.runs_dir / "error_analysis" / "error_summary.json"
        deployed_pt = self.artifacts_dir / "model.pt"
        final_test_report = self.runs_dir / "evaluate" / "final_test_report.json"

        return {
            "data": StageDefinition(
                name="data",
                description="Ingest, Harmonize, Dedup (SHA+pHash), Group Split & Quality Gates",
                inputs=[Path(p) for p in DEFAULT_ARCHIVES],
                outputs=[data_yaml, manifest_csv, audit_json],
                runner=self._run_data_stage,
            ),
            "train": StageDefinition(
                name="train",
                description="Train YOLOv8 baseline with fixed seed & metadata tracking",
                inputs=[data_yaml, self.config_path],
                outputs=[best_weights],
                runner=self._run_train_stage,
            ),
            "evaluate": StageDefinition(
                name="evaluate",
                description="Evaluate on Validation set (mAP50-95, mAP50, Recall, AP per class)",
                inputs=[best_weights, data_yaml],
                outputs=[experiments_csv],
                runner=self._run_evaluate_stage,
            ),
            "compare": StageDefinition(
                name="compare",
                description="Legacy report only; không thuộc canonical pipeline",
                inputs=[experiments_csv],
                outputs=[],
                runner=self._run_compare_stage,
            ),
            "errors": StageDefinition(
                name="errors",
                description="Analyze errors by lesion size (S/M/L) and taxonomy (FP/FN/Overlap)",
                inputs=[best_weights, data_yaml],
                outputs=[error_summary],
                runner=self._run_error_stage,
            ),
            "test": StageDefinition(
                name="test",
                description="Execute locked final test evaluation protocol (one-time report)",
                inputs=[best_weights, data_yaml],
                outputs=[final_test_report],
                runner=self._run_test_stage,
                requires_confirmation=True,
            ),
            "export": StageDefinition(
                name="export",
                description="Export weights to ONNX/TorchScript and verify prediction parity",
                inputs=[best_weights],
                outputs=[deployed_pt],
                runner=self._run_export_stage,
            ),
        }

    def dry_run(self) -> None:
        """In sơ đồ DAG và kiểm tra điều kiện tiên quyết cho từng stage mà không chạy."""
        stages = self.get_stages()
        print("\n" + "=" * 80)
        print("  RICE LEAF DISEASE RECOGNITION - PIPELINE EXECUTION PLAN (DRY-RUN)")
        print("=" * 80)

        for name, stage in stages.items():
            print(f"\n[Stage: {name.upper()}] - {stage.description}")
            missing_inputs = [p for p in stage.inputs if not p.exists()]
            existing_outputs = [p for p in stage.outputs if p.exists()]

            print(f"  Inputs required : {len(stage.inputs)}")
            for inp in stage.inputs:
                status_mark = "✓ EXISTS" if inp.exists() else "✗ MISSING"
                print(f"    - {inp.as_posix():<55} [{status_mark}]")

            print(f"  Outputs target  : {len(stage.outputs)}")
            for out in stage.outputs:
                status_mark = "✓ READY" if out.exists() else "○ PENDING"
                print(f"    - {out.as_posix():<55} [{status_mark}]")

            if stage.requires_confirmation:
                print("  Note: Requires explicit confirmation flag (--confirm-final-test)")

            if missing_inputs:
                print(f"  Status: BLOCKED (Missing {len(missing_inputs)} prerequisite inputs)")
            elif len(existing_outputs) == len(stage.outputs) and stage.outputs:
                print("  Status: UP-TO-DATE (Artifacts already exist, use --force to rerun)")
            else:
                print("  Status: READY TO EXECUTE")

        print("\n" + "=" * 80 + "\n")

    def _run_data_stage(self, force: bool = False) -> list[Path]:
        from .deduplication import deduplicate_and_group
        from .prepare import (
            assign_splits,
            collect_records,
            prepare_sources,
            write_dataset,
        )

        data_yaml = self.output_dir / "data.yaml"
        manifest_csv = self.output_dir / "manifest.csv"
        audit_json = self.output_dir / "audit_report.json"

        if data_yaml.exists() and not force:
            logger.info("Stage 'data': Artifacts đã tồn tại tại %s, bỏ qua.", self.output_dir)
            return [data_yaml, manifest_csv, audit_json]

        seed_everything(SEED)
        extract_dir = Path("data/extracted")
        sources = prepare_sources([Path(p) for p in DEFAULT_ARCHIVES], extract_dir)
        records, audit = collect_records(sources, keep_negatives=True)
        records, dedup_audit = deduplicate_and_group(records, phash_distance=2)
        audit.update(dedup_audit)

        assign_splits(records)
        write_dataset(records, audit, self.output_dir, overwrite=True)
        return [data_yaml, manifest_csv, audit_json]

    def _run_train_stage(self, epochs: int | None = None, force: bool = False) -> list[Path]:
        if self.config is None:
            raise RuntimeError("Không thể train khi config không được nạp")
        architecture = self.config.model.architecture
        train_run_dir = self.runs_dir / "train" / f"{architecture}_640"
        best_weights = train_run_dir / "weights" / "best.pt"

        if best_weights.exists() and not force:
            logger.info("Stage 'train': Trọng số tốt nhất đã có tại %s, bỏ qua.", best_weights)
            return [best_weights]

        import torch
        from ultralytics import YOLO

        data_yaml = self.output_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Thiếu {data_yaml}. Hãy chạy stage 'data' trước.")

        device = "0" if torch.cuda.is_available() else "cpu"
        train_epochs = epochs if epochs is not None else self.config.training.epochs
        batch = (
            self.config.training.batch_gpu
            if torch.cuda.is_available()
            else self.config.training.batch_cpu
        )
        augmentation = self.config.training.augmentation

        model = YOLO(self.config.model.weights)
        model.train(
            data=str(data_yaml),
            epochs=train_epochs,
            batch=batch,
            imgsz=self.config.data.image_size,
            device=device,
            optimizer=self.config.training.optimizer,
            lr0=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            patience=self.config.training.patience,
            hsv_h=augmentation.hsv_h,
            hsv_s=augmentation.hsv_s,
            hsv_v=augmentation.hsv_v,
            degrees=augmentation.degrees,
            translate=augmentation.translate,
            scale=augmentation.scale,
            fliplr=augmentation.fliplr,
            flipud=augmentation.flipud,
            mosaic=augmentation.mosaic,
            mixup=augmentation.mixup,
            close_mosaic=augmentation.close_mosaic,
            seed=SEED,
            deterministic=True,
            workers=0 if sys.platform == "win32" else 2,
            val=True,
            save=True,
            project=str(self.runs_dir / "train"),
            name=f"{architecture}_640",
            exist_ok=True,
            verbose=False,
        )

        if not best_weights.exists():
            # Fallback nếu epoch quá ngắn
            last_weights = train_run_dir / "weights" / "last.pt"
            if last_weights.exists():
                shutil.copy2(last_weights, best_weights)

        # Ghi metadata truy vết MLOps
        metadata = {
            "run_name": f"{architecture}_640",
            "data_yaml": str(data_yaml.resolve()),
            "data_manifest_sha256": sha256_file(self.output_dir / "manifest.csv"),
            "best_weights_sha256": sha256_file(best_weights),
            "seed": SEED,
            "epochs": train_epochs,
            "architecture": architecture,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (train_run_dir / "run_metadata.json").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )
        return [best_weights]

    def _find_best_weights(self) -> Path:
        """Lấy đúng artifact của run canonical, không quét ``best.pt`` ngẫu nhiên."""
        architecture = self.config.model.architecture if self.config else "yolov8s"
        canonical = self.runs_dir / "train" / f"{architecture}_640" / "weights" / "best.pt"
        if canonical.exists():
            return canonical
        release_model = self.artifacts_dir / "model.pt"
        if release_model.exists():
            return release_model
        raise FileNotFoundError(
            f"Không tìm thấy trọng số canonical tại {canonical}. Hãy chạy stage 'train' trước."
        )

    def _run_evaluate_stage(self, force: bool = False) -> list[Path]:
        from ultralytics import YOLO

        best_weights = self._find_best_weights()
        data_yaml = self.output_dir / "data.yaml"
        output_eval = self.runs_dir / "evaluate"
        output_eval.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(best_weights))
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=640,
            batch=4,
            conf=0.001,
            iou=0.7,
            plots=True,
            project=str(output_eval),
            name="val_evaluation",
            exist_ok=True,
            verbose=False,
        )

        summary = {
            "run_name": best_weights.parent.parent.name,
            "weights": str(best_weights.resolve()),
            "split": "val",
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        experiments_csv = output_eval / "experiments.csv"
        current_df = pd.DataFrame([summary])
        if experiments_csv.exists():
            history_df = pd.read_csv(experiments_csv)
            history_df = history_df[
                ~((history_df["run_name"] == summary["run_name"]) & (history_df["split"] == "val"))
            ]
            current_df = pd.concat([history_df, current_df], ignore_index=True)
        current_df.to_csv(experiments_csv, index=False)

        logger.info("Hoàn thành đánh giá Validation. mAP50-95: %.4f", summary["mAP50-95"])
        return [experiments_csv]

    def _run_compare_stage(self) -> list[Path]:
        experiments_csv = self.runs_dir / "evaluate" / "experiments.csv"
        if not experiments_csv.exists():
            raise FileNotFoundError(f"Chưa có {experiments_csv}. Hãy chạy stage 'evaluate' trước.")

        df = pd.read_csv(experiments_csv)
        val_df = df[df["split"] == "val"].sort_values("mAP50-95", ascending=False)
        if val_df.empty:
            raise ValueError("Không có kết quả xác thực (split=val) trong experiments.csv")

        print("\n--- BẢNG XẾP HẠNG MÔ HÌNH (VALIDATION-ONLY SELECTION) ---")
        cols = ["run_name", "precision", "recall", "mAP50", "mAP50-95"]
        print(val_df[cols].to_string(index=False))
        champion = val_df.iloc[0]["run_name"]
        print(f"\n=> CHAMPION MODEL ĐỀ XUẤT: {champion}")
        print("Lưu ý: Tập Test hoàn toàn không được dùng để xếp hạng nhằm chống rò rỉ dữ liệu.\n")
        return [experiments_csv]

    def _run_error_stage(self) -> list[Path]:
        from .error_analysis import run_error_analysis

        best_weights = self._find_best_weights()
        data_yaml = self.output_dir / "data.yaml"
        output_errors = self.runs_dir / "error_analysis"

        summary = run_error_analysis(
            weights_path=best_weights,
            data_yaml_path=data_yaml,
            split="val",
            confidence=self.config.policy.review_threshold if self.config else 0.20,
            output_dir=output_errors,
        )
        logger.info("Hoàn thành phân tích lỗi: %s", summary.get("total_predictions", 0))
        return [output_errors / "error_summary.json"]

    def _run_test_stage(
        self,
        confirmed: bool = False,
        force_reopen: bool = False,
    ) -> list[Path]:
        if not confirmed:
            raise PermissionError(
                "Tập Test bị khóa. Hãy thêm cờ '--confirm-final-test' sau khi đã "
                "khóa model và policy."
            )

        from ultralytics import YOLO

        best_weights = self._find_best_weights()
        data_yaml = self.output_dir / "data.yaml"
        output_eval = self.runs_dir / "evaluate"
        final_report = output_eval / "final_test_report.json"
        if final_report.exists() and not force_reopen:
            raise FileExistsError(
                f"Final test đã tồn tại tại {final_report}. "
                "Dùng --force-reopen-test nếu thật sự cần mở lại; báo cáo sẽ bị "
                "đánh dấu compromised."
            )

        model = YOLO(str(best_weights))
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=640,
            batch=4,
            conf=0.001,
            iou=0.7,
            plots=True,
            project=str(output_eval),
            name="test_final_evaluation",
            exist_ok=True,
            verbose=False,
        )

        test_summary = {
            "run_name": best_weights.parent.parent.name,
            "weights": str(best_weights.resolve()),
            "split": "test",
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_hash": sha256_file(best_weights),
            "test_compromised": bool(final_report.exists() and force_reopen),
        }

        manifest_path = self.output_dir / "manifest.csv"
        data_manifest_path = self.output_dir / "data_manifest.json"
        test_summary["dataset_manifest_sha256"] = sha256_file(manifest_path)
        if data_manifest_path.exists():
            data_manifest = yaml.safe_load(data_manifest_path.read_text(encoding="utf-8")) or {}
            test_summary["test_split_hash"] = data_manifest.get("split_hashes", {}).get("test")
            test_summary["dataset_id"] = data_manifest.get("dataset_id")

        # Final test là artifact bất biến; không trộn vào lịch sử dùng để chọn model.
        final_report.write_text(
            json.dumps(
                {
                    "model_version": "unreleased",
                    "metrics": {
                        "map50_95": test_summary["mAP50-95"],
                        "map50": test_summary["mAP50"],
                        "precision": test_summary["precision"],
                        "recall": test_summary["recall"],
                    },
                    **test_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\n--- KẾT QUẢ ĐÁNH GIÁ TẬP TEST CHÍNH THỨC (FINAL TEST METRICS) ---")
        print(pd.DataFrame([test_summary]).to_string(index=False))
        return [final_report]

    def _run_export_stage(self) -> list[Path]:
        from .export import export_model, verify_prediction_parity

        best_weights = self._find_best_weights()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sao chép trọng số canonical sang model.pt; không dùng tên chung
        # best.pt để tránh runtime nạp nhầm model của run khác.
        target_pt = self.artifacts_dir / "model.pt"
        if best_weights.resolve() != target_pt.resolve():
            shutil.copy2(best_weights, target_pt)

        # 2. Xuất mô hình sang ONNX (hoặc TorchScript fallback)
        export_meta = export_model(
            weights_path=target_pt,
            target_format="onnx",
            imgsz=640,
            output_dir=self.artifacts_dir,
        )
        target_exported = Path(export_meta["exported_file"])

        # 3. Kiểm định Prediction Parity nếu có ảnh mẫu
        sample_images = list(Path("data/sample").glob("*.jpg"))
        if sample_images:
            parity_report = verify_prediction_parity(
                pytorch_model_path=target_pt,
                exported_model_path=target_exported,
                sample_images=sample_images,
            )
            print(
                f"\nPrediction Parity check ({export_meta['format']}): "
                f"Passed={parity_report['parity_passed']}, "
                f"Max Conf Diff={parity_report['max_conf_diff']:.4f}"
            )

        if self.config is not None:
            policy_path = self.artifacts_dir / "detection_policy.json"
            write_json(
                policy_path,
                {
                    "model_version": "unreleased",
                    "review_threshold": self.config.policy.review_threshold,
                    "accept_threshold": self.config.policy.accept_threshold,
                    "candidate_confidence": self.config.inference.candidate_confidence,
                    "iou": self.config.inference.iou,
                },
            )
            write_json(
                self.artifacts_dir / "model_metadata.json",
                {
                    "model_version": "unreleased",
                    "architecture": self.config.model.architecture,
                    "image_size": self.config.data.image_size,
                    "classes": CLASS_NAMES,
                    "weights_sha256": sha256_file(target_pt),
                    "training_config_sha256": sha256_file(self.config_path),
                    "dataset_manifest_sha256": (
                        sha256_file(self.output_dir / "manifest.csv")
                        if (self.output_dir / "manifest.csv").exists()
                        else None
                    ),
                    "policy_sha256": sha256_file(policy_path),
                    "policy_file": str(policy_path.resolve()),
                },
            )

        return [target_pt, target_exported]

    def execute_stage(
        self,
        stage_name: str,
        force: bool = False,
        epochs: int | None = None,
        confirm_final_test: bool = False,
        force_reopen_test: bool = False,
    ) -> PipelineStatus:
        """Thực thi một stage duy nhất và kiểm tra lỗi."""
        stages = self.get_stages()
        if stage_name not in stages:
            raise ValueError(f"Stage không hợp lệ: '{stage_name}'. Hỗ trợ: {list(stages.keys())}")

        stage = stages[stage_name]
        logger.info(">>> BẮT ĐẦU STAGE [%s]: %s", stage.name.upper(), stage.description)
        start_time = time.time()

        try:
            # Kiểm tra inputs tiên quyết
            missing_inputs = [inp for inp in stage.inputs if not inp.exists()]
            if missing_inputs:
                missing_str = ", ".join(p.as_posix() for p in missing_inputs)
                raise FileNotFoundError(
                    f"Stage '{stage_name}' thiếu inputs tiên quyết: [{missing_str}]. "
                    "Hãy chạy các stage phía trước trước."
                )

            # Thực thi runner tương ứng
            if stage_name == "data":
                produced = stage.runner(force=force)
            elif stage_name == "train":
                produced = stage.runner(epochs=epochs, force=force)
            elif stage_name == "evaluate":
                produced = stage.runner(force=force)
            elif stage_name == "test":
                produced = stage.runner(
                    confirmed=confirm_final_test,
                    force_reopen=force_reopen_test,
                )
            else:
                produced = stage.runner()

            elapsed = time.time() - start_time
            logger.info("<<< HOÀN THÀNH STAGE [%s] trong %.2f giây", stage.name.upper(), elapsed)
            return PipelineStatus(
                stage_name=stage.name,
                status="SUCCESS",
                duration_sec=elapsed,
                message="Completed successfully",
                produced_artifacts=produced or [],
            )

        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error("!!! THẤT BẠI STAGE [%s]: %s", stage.name.upper(), exc)
            return PipelineStatus(
                stage_name=stage.name,
                status="FAILED",
                duration_sec=elapsed,
                message=str(exc),
            )

    def run_all(
        self,
        force: bool = False,
        epochs: int | None = None,
        confirm_final_test: bool = False,
        force_reopen_test: bool = False,
    ) -> list[PipelineStatus]:
        """Chạy toàn bộ quy trình pipeline theo thứ tự chuẩn."""
        # compare là tool legacy, không nằm trong đường chạy canonical.
        sequence = ["data", "train", "evaluate", "errors"]
        if confirm_final_test:
            sequence.append("test")
        sequence.append("export")

        results = []
        for stg in sequence:
            status = self.execute_stage(
                stage_name=stg,
                force=force,
                epochs=epochs,
                confirm_final_test=confirm_final_test,
                force_reopen_test=force_reopen_test,
            )
            results.append(status)
            if status.status == "FAILED":
                logger.error("Pipeline bị gián đoạn tại stage '%s'.", stg)
                break

        self._print_summary(results)
        return results

    def _print_summary(self, results: list[PipelineStatus]) -> None:
        print("\n" + "=" * 80)
        print("          PIPELINE EXECUTION SUMMARY REPORT")
        print("=" * 80)
        print(f"{'Stage':<15} | {'Status':<10} | {'Duration':<12} | {'Message'}")
        print("-" * 80)
        for res in results:
            stage_col = f"{res.stage_name:<15}"
            status_col = f"{res.status:<10}"
            dur_col = f"{res.duration_sec:>8.2f}s"
            print(f"{stage_col} | {status_col} | {dur_col}    | {res.message}")
        print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bộ điều phối Pipeline Nhận Diện Bệnh Lá Lúa (Pipeline Orchestrator)"
    )
    parser.add_argument(
        "--stage",
        choices=["data", "train", "evaluate", "compare", "errors", "test", "export", "all"],
        default="all",
        help="Chọn stage cần thực thi (mặc định: 'all')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In sơ đồ DAG và điều kiện tiên quyết mà không chạy",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Đường dẫn file cấu hình YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Số epoch huấn luyện khi chạy stage train",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Chạy lại stage và ghi đè artifact cũ nếu đã tồn tại",
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Xác nhận mở khóa đánh giá tập Test chính thức",
    )
    parser.add_argument(
        "--force-reopen-test",
        action="store_true",
        help="Mở lại final test đã có báo cáo và đánh dấu test_compromised=true",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    orchestrator = PipelineOrchestrator(config_path=args.config)

    if args.dry_run:
        orchestrator.dry_run()
        return

    if args.stage == "all":
        orchestrator.run_all(
            force=args.force,
            epochs=args.epochs,
            confirm_final_test=args.confirm_final_test,
            force_reopen_test=args.force_reopen_test,
        )
    else:
        status = orchestrator.execute_stage(
            stage_name=args.stage,
            force=args.force,
            epochs=args.epochs,
            confirm_final_test=args.confirm_final_test,
            force_reopen_test=args.force_reopen_test,
        )
        orchestrator._print_summary([status])


if __name__ == "__main__":
    main()
