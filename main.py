from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

FS_PATH = os.getenv("FS_PATH")
BASE_DIR = os.path.join(FS_PATH, "flux_train")
# Directory to save training runs
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

class DatasetConfig(BaseModel):
    output_name: str
    num_repeats: int = 10
    resolution: int = 1024
    batch_size: int = 1
    shuffle_caption: bool = False
    caption_extension: str = ".txt"
    keep_tokens: int = 1

@app.post("/create-dataset-config")
def create_dataset_config(config: DatasetConfig):
    config_path = os.path.join(DATASETS_DIR, f"{config.output_name}.toml")
    
    toml_content = f"""[general]
shuffle_caption = {str(config.shuffle_caption).lower()}
caption_extension = '{config.caption_extension}'
keep_tokens = {config.keep_tokens}

[[datasets]]
resolution = {config.resolution}
batch_size = {config.batch_size}
keep_tokens = {config.keep_tokens}

[[datasets.subsets]]
image_dir = '{DATASETS_DIR}/{config.output_name}'
class_tokens = '{config.output_name}'
num_repeats = {config.num_repeats}
"""
    
    try:
        with open(config_path, "w") as f:
            f.write(toml_content)
        return {"message": "Dataset config created", "path": config_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TrainRequest(BaseModel):
    output_name: str
    pretrained_model: str = 'flux1-dev.sft'
    clip_l: str = 'clip_l.safetensors'
    t5xxl: str = 't5xxl_fp16.safetensors'
    ae: str = 'ae.sft'
    max_train_epochs: int = 10
    learning_rate: float = 8e-4
    network_dim: int = 4
    save_every_n_epochs: int = 1
    enable_bucket: bool = True
    full_bf16: bool = True

@app.post("/train")
def train_lora(request: TrainRequest):
    output_dir = os.path.join(OUTPUTS_DIR, request.output_name)
    os.makedirs(output_dir, exist_ok=True)

    command = f"""accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 sd_scripts/flux_train_network.py \
--pretrained_model_name_or_path {MODELS_DIR}/{request.pretrained_model} --clip_l {MODELS_DIR}/{request.clip_l} --t5xxl {MODELS_DIR}/{request.t5xxl} \
--ae {request.ae} --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
--max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
--network_module networks.lora_flux --network_dim {request.network_dim} --network_train_unet_only \
--optimizer_type adamw8bit --learning_rate {request.learning_rate} \
--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
--highvram --max_train_epochs {request.max_train_epochs} --save_every_n_epochs {request.save_every_n_epochs} --dataset_config {DATASETS_DIR}/{request.output_name}.toml \
--output_dir {output_dir} --output_name {request.output_name} \
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 --loss_type l2 {"--enable_bucket" if request.enable_bucket else ""} {"--full_bf16" if request.full_bf16 else ""}
"""

    try:
        print("Running command")
        print(command)
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"message": "Training started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{name}")
def get_training_status(name: str):
    log_file = os.path.join(OUTPUTS_DIR, name, "train.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return {"name": name, "status": f.readlines()[-10:]}  # Return last 10 log lines
    return {"name": name, "status": "Not found or still running"}