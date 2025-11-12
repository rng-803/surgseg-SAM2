"""Convert ONNX refiner into a TensorRT engine."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - optional dependency
    trt = None


def build_engine(args: argparse.Namespace) -> None:
    if trt is None:
        raise ImportError("TensorRT is not installed. Please install it before running this script.")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    onnx_path = str(args.onnx)
    with open(onnx_path, "rb") as fp:
        if not parser.parse(fp.read()):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise RuntimeError("Failed to parse ONNX model")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 20))
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(engine)
    print(f"TensorRT engine saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("export/refiner.onnx"))
    parser.add_argument("--output", type=Path, default=Path("export/refiner_fp16.plan"))
    parser.add_argument("--workspace", type=int, default=1024, help="Workspace size in MB")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
