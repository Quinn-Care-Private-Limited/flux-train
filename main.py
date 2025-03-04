from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import uuid
import os

app = FastAPI()

# Directory to save training runs
TRAINING_DIR = "outputs"
os.makedirs(TRAINING_DIR, exist_ok=True)

class TrainRequest(BaseModel):
    pretrained_model: str
    clip_l: str
    t5xxl: str
    ae: str
    dataset_config: str
    output_dir: str
    output_name: str
    max_train_epochs: int = 4
    learning_rate: float = 1e-4
    network_dim: int = 4

@app.post("/train")
def train_lora(request: TrainRequest):
    run_id = str(uuid.uuid4())  # Unique ID for this training session
    output_dir = os.path.join(TRAINING_DIR, request.output_name)
    os.makedirs(output_dir, exist_ok=True)

    command = f"""
    accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py 
    --pretrained_model_name_or_path {request.pretrained_model} --clip_l {request.clip_l} --t5xxl {request.t5xxl} 
    --ae {request.ae} --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers 
    --max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 
    --network_module networks.lora_flux --network_dim {request.network_dim} --network_train_unet_only 
    --optimizer_type adamw8bit --learning_rate {request.learning_rate} 
    --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base 
    --highvram --max_train_epochs {request.max_train_epochs} --save_every_n_epochs 1 --dataset_config {request.dataset_config} 
    --output_dir {output_dir} --output_name {request.output_name} 
    --timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0
    """

    try:
        subprocess.Popen(command, shell=True, cwd="sd-scripts", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"message": "Training started", "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{run_id}")
def get_training_status(run_id: str):
    log_file = os.path.join(TRAINING_DIR, run_id, "train.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return {"run_id": run_id, "status": f.readlines()[-10:]}  # Return last 10 log lines
    return {"run_id": run_id, "status": "Not found or still running"}