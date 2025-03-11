# Example Flask API to interact with the server manager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataclasses import asdict
from job_manager import ServerManager
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
port = int(os.environ.get("PORT", "8080"))
server_manager = ServerManager()

class JobRequest(BaseModel):
    output_name: str
    trigger_word: str
    num_repeats: int = 10
    resolution: int = 1024
    pretrained_model: str = 'flux1-dev-fp8.sft'
    clip_l: str = 'clip_l.safetensors'
    t5xxl: str = 't5xxl_fp16.safetensors'
    ae: str = 'ae.sft'
    max_train_epochs: int = 10
    learning_rate: float = 8e-4
    network_dim: int = 4
    save_every_n_epochs: int = 1
    enable_bucket: bool = True
    full_bf16: bool = False


@app.post("/run")
def run_job(request: JobRequest):
    """Run a new job to the queue"""
    job_id = server_manager.run_job(request.model_dump())
    return {"job_id": job_id, "status": "submitted"}


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Get the status of a job"""
    status = server_manager.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status


@app.get("/servers")
def get_servers_status():
    """Get the status of all servers"""
    with server_manager.server_lock:
        servers = {name: asdict(server) for name, server in server_manager.servers.items()}
    
    return servers


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


# Main entry point
if __name__ == "__main__":
    import uvicorn

    try:
        # Start the FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        server_manager.shutdown()