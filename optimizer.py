import os
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def get_model_info(model_path: str) -> dict:
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.random.randn(*shape).astype(np.float32)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "size_mb": round(size_mb, 2),
        "latency_ms": round(np.median(times), 2),
        "latency_p95_ms": round(np.percentile(times, 95), 2),
        "input_name": inp.name,
        "input_shape": shape,
    }


def optimize_model(input_path: str, output_path: str, strategy: str = "int8") -> dict:
    before = get_model_info(input_path)

    quant_type = QuantType.QInt8
    if strategy == "int4":
        quant_type = QuantType.QUInt4

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=quant_type,
    )

    after = get_model_info(output_path)

    size_reduction = round(before["size_mb"] / after["size_mb"], 1) if after["size_mb"] > 0 else 0
    speedup = round(before["latency_ms"] / after["latency_ms"], 1) if after["latency_ms"] > 0 else 0
    size_saved_pct = round((1 - after["size_mb"] / before["size_mb"]) * 100, 1) if before["size_mb"] > 0 else 0

    return {
        "before": before,
        "after": after,
        "delta": {
            "size_reduction": f"{size_reduction}x",
            "speedup": f"{speedup}x",
            "size_saved_pct": f"{size_saved_pct}%",
        },
        "techniques_applied": [f"dynamic_{strategy}_quantization"],
    }