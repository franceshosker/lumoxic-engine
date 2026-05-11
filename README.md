# Lumoxic Engine

The optimization engine behind [Lumoxic AI](https://lumoxicai.me) — a FastAPI service that quantizes, prunes, and benchmarks ONNX models.

## Features

- **Dynamic INT8/INT4 Quantization** — Post-training quantization via ONNX Runtime
- **Graph Optimization** — Operator fusion, constant folding, dead node elimination
- **Benchmarking** — Per-inference latency, energy, and throughput measurement
- **REST API** — Upload a model, get back an optimized version

## Quick Start

```bash
pip install -r requirements.txt
python api.py
```

Server starts on `http://0.0.0.0:8081`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/optimize` | Upload .onnx model for optimization |
| GET | `/v1/jobs/{id}` | Check job status |
| GET | `/v1/jobs/{id}/result` | Get optimization results |
| GET | `/v1/models/{id}/download` | Download optimized model |
| POST | `/v1/benchmark` | Benchmark without optimizing |
| GET | `/v1/health` | Health check |

## Stack

- **FastAPI** + **Uvicorn**
- **ONNX Runtime** (quantization + inference)
- **NumPy** (benchmarking)

## License

All rights reserved. © 2026 Lumoxic AI.