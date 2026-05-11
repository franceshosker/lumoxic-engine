import os
import uuid
import shutil
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from optimizer import optimize_model, get_model_info

app = FastAPI(title="Lumoxic AI Optimization Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = "/tmp/lumoxic_jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

jobs: dict = {}


@app.get("/v1/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/v1/optimize")
async def optimize(
    model: UploadFile = File(...),
    strategy: str = Form("int8"),
    target: str = Form("server"),
):
    if not model.filename or not model.filename.endswith(".onnx"):
        raise HTTPException(400, "Only .onnx models supported")

    if strategy not in ("int8", "int4", "auto"):
        raise HTTPException(400, "Strategy must be int8, int4, or auto")

    job_id = str(uuid.uuid4())[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir)

    input_path = os.path.join(job_dir, "input.onnx")
    output_path = os.path.join(job_dir, "optimized.onnx")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(model.file, f)

    size_mb = os.path.getsize(input_path) / (1024 * 1024)
    if size_mb > 500:
        shutil.rmtree(job_dir)
        raise HTTPException(413, "Model too large (max 500MB)")

    jobs[job_id] = {"status": "running", "created": time.time()}

    try:
        if strategy == "auto":
            strategy = "int8"

        result = optimize_model(input_path, output_path, strategy)

        jobs[job_id] = {
            "status": "completed",
            "job_id": job_id,
            "model": model.filename,
            "target": target,
            "optimization": {
                "strategy": strategy,
                "techniques_applied": result["techniques_applied"],
            },
            "before": result["before"],
            "after": result["after"],
            "delta": result["delta"],
            "download_url": f"/v1/models/{job_id}/download",
            "created": jobs[job_id]["created"],
            "completed": time.time(),
        }

        return jobs[job_id]

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}
        raise HTTPException(500, f"Optimization failed: {e}")


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/v1/jobs/{job_id}/result")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] != "completed":
        raise HTTPException(400, f"Job status: {jobs[job_id]['status']}")
    return jobs[job_id]


@app.get("/v1/models/{job_id}/download")
def download_model(job_id: str):
    path = os.path.join(JOBS_DIR, job_id, "optimized.onnx")
    if not os.path.exists(path):
        raise HTTPException(404, "Model not found")
    return FileResponse(path, media_type="application/octet-stream", filename=f"optimized_{job_id}.onnx")


@app.post("/v1/benchmark")
async def benchmark(model: UploadFile = File(...)):
    if not model.filename or not model.filename.endswith(".onnx"):
        raise HTTPException(400, "Only .onnx models supported")

    tmp_path = f"/tmp/bench_{uuid.uuid4().hex[:8]}.onnx"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(model.file, f)

    try:
        info = get_model_info(tmp_path)
        return {"model": model.filename, "benchmark": info}
    finally:
        os.remove(tmp_path)


@app.get("/v1/usage")
def usage():
    completed = sum(1 for j in jobs.values() if j.get("status") == "completed")
    failed = sum(1 for j in jobs.values() if j.get("status") == "failed")
    return {"total_jobs": len(jobs), "completed": completed, "failed": failed}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)