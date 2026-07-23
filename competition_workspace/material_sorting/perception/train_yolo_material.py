import argparse
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PERCEPTION_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PERCEPTION_DIR / "checkpoints" / "material_box.pt"


def import_yolo():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    (ROOT / ".ultralytics").mkdir(parents=True, exist_ok=True)
    if str(PERCEPTION_DIR) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_DIR))
    from backends import _load_ultralytics_yolo

    return _load_ultralytics_yolo()


def train(args):
    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO base model not found: {model_path}. "
            "Put yolov8n.pt/yolo11n.pt in the repo or use the existing material_box.pt."
        )

    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project
    project.mkdir(parents=True, exist_ok=True)

    YOLO = import_yolo()
    model = YOLO(str(model_path))
    result = model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        project=str(project),
        name=str(args.name),
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    save_dir = Path(getattr(result, "save_dir", project / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    export_path = None
    if args.export_checkpoint:
        export_path = Path(args.export_checkpoint)
        if not export_path.is_absolute():
            export_path = ROOT / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        source_pt = best_pt if best_pt.is_file() else last_pt
        if source_pt.is_file():
            shutil.copy2(source_pt, export_path)

    summary = {
        "data_yaml": data_yaml.as_posix(),
        "base_model": model_path.as_posix(),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "save_dir": save_dir.as_posix(),
        "best_pt": best_pt.as_posix() if best_pt.is_file() else None,
        "last_pt": last_pt.as_posix() if last_pt.is_file() else None,
        "export_checkpoint": None if export_path is None else export_path.as_posix(),
    }
    summary_path = save_dir / "material_yolo_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[train] save_dir={save_dir}")
    print(f"[train] best_pt={summary['best_pt']}")
    print(f"[train] export_checkpoint={summary['export_checkpoint']}")
    print(f"[train] summary={summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train/fine-tune YOLO for material sorting boxes.")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default=str(ROOT / "reports" / "material_sorting_yolo_training"))
    parser.add_argument("--name", default="yellow_demo")
    parser.add_argument(
        "--export-checkpoint",
        default=str(PERCEPTION_DIR / "checkpoints" / "material_box_yellow_demo.pt"),
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
