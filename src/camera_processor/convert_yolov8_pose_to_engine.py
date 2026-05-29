import argparse
from pathlib import Path
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a YOLOv8 pose .pt model to TensorRT .engine in the same models folder.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "yolov8n-pose.pt",
        help="Path to the source .pt model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference/export image size.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for export: 'auto', CUDA id like '0', or 'cpu'.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        return 1

    if model_path.suffix.lower() != ".pt":
        print(f"ERROR: expected a .pt file, got: {model_path.name}")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("ERROR: could not import ultralytics. Install it first.")
        print(f"Details: {exc}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"

    if args.device == "cpu":
        print("ERROR: TensorRT (.engine) export needs CUDA/TensorRT. No CUDA device was selected.")
        print("Tip: run this script in a machine/container with NVIDIA GPU and CUDA enabled.")
        return 1

    print("Exporting to TensorRT engine...")
    print(f"Settings -> imgsz={args.imgsz}, half={args.half}, device={args.device}")

    try:
        exported = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=args.half,
            device=args.device,
        )
    except Exception as exc:
        print("ERROR: export failed.")
        print("TensorRT export usually requires NVIDIA GPU, CUDA, TensorRT, and a compatible PyTorch/Ultralytics setup.")
        print(f"Details: {exc}")
        return 1

    # Ultralytics returns the exported path as str/Path in most versions.
    exported_path = Path(str(exported)).resolve()
    print(f"Export complete: {exported_path}")

    if exported_path.parent != model_path.parent:
        print("Note: exported file was not generated in the same folder as the source model.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
