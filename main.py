from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from io import BytesIO
import subprocess
import uuid
import os
import torch
import requests
import asyncio
import gpustat
import psutil

from captioner import caption_images_in_directory
app = FastAPI()

FS_PATH = os.getenv("FS_PATH")
BASE_DIR = os.path.join(FS_PATH, "flux_train")
# Directory to save training runs
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)


@app.get("/health")
def get_health():
    gpu_stats = gpustat.new_query()
    return {"status": "ok", "gpu": gpu_stats.gpus[0].utilization, "cpu": psutil.cpu_percent(interval=1)}
class DownloadRequest(BaseModel):
    urls: list[str]
    output_name: str
    captions: list[str] = []

@app.post("/download-images")
def download_images(request: DownloadRequest):
    dataset_dir = os.path.join(DATASETS_DIR, request.output_name)
    os.makedirs(dataset_dir, exist_ok=True)

    saved_files = {}
    for url in request.urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise error for bad responses

            # Extract filename from URL
            filename = os.path.basename(url.split("?")[0])  # Remove URL parameters
            if not filename.lower().endswith(("jpg", "jpeg", "png")):
                filename += ".jpg"  # Default to JPG if no extension

            file_path = os.path.join(dataset_dir, filename)

            # Save image
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(file_path, format="JPEG")  # Save as JPG format

            # Save caption if provided
            if len(request.captions):
                caption_path = os.path.splitext(file_path)[0] + ".txt"  # Save as .txt with same name
                with open(caption_path, "w") as f:
                    f.write(request.captions.pop(0))

            saved_files[url] = file_path
        except Exception as e:
            saved_files[url] = f"Error: {str(e)}"

    return {"message": "Download complete", "files": saved_files}


class CaptionRequest(BaseModel):
    output_name: str  # Path to directory containing images
    trigger_word: str
    prompt: str = "Describe this image in detail."  # Default prompt

@app.post("/caption-images")
def caption_images(request: CaptionRequest):
    # Load Florence-2 model and processor
    MODEL_PATH = f"{MODELS_DIR}/florence2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    dataset_dir = os.path.join(DATASETS_DIR, request.output_name)
    if not os.path.exists(dataset_dir):
        raise HTTPException(status_code=400, detail="Image directory not found")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    captions = {}

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(("jpg", "jpeg", "png")):
            image_path = os.path.join(dataset_dir, filename)
            caption_path = os.path.splitext(image_path)[0] + ".txt"  # Save as .txt with same name

            try:
                # Load image
                image = Image.open(image_path).convert("RGB")

                # Process image and generate caption
                inputs = processor(text=request.prompt, images=image, return_tensors="pt").to(device, torch_dtype)
                outputs = model.generate(**inputs, max_new_tokens=1024, num_beams=3)
                caption: str = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                caption = caption.replace("The image shows ", "")
                caption = f"{request.trigger_word}, {caption}"

                # Save caption to a .txt file
                with open(caption_path, "w") as f:
                    f.write(caption)

                captions[filename] = caption
            except Exception as e:
                captions[filename] = f"Error: {str(e)}"

    model.to("cpu")
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"message": "Captions generated and saved", "captions": captions}

class DatasetConfig(BaseModel):
    output_name: str
    trigger_word: str
    num_repeats: int = 1
    resolution: int = 1024
    batch_size: int = 1
    pretrained_model: str = 'flux1-dev-fp8.sft'
    clip_l: str = 'clip_l.safetensors'
    t5xxl: str = 't5xxl_fp16.safetensors'
    ae: str = 'ae.sft'

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
class_tokens = '{config.trigger_word}'
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
    image_urls: list[str] = []
    captions: list[str] = []
    auto_captioning: bool = False
    steps: int = 500
    resolution: int = 1024
    learning_rate: float = 8e-4
    network_dim: int = 4
    enable_bucket: bool = True
    full_bf16: bool = True
    pretrained_model: str = 'flux1-dev-fp8.sft'
    clip_l: str = 'clip_l.safetensors'
    t5xxl: str = 't5xxl_fp16.safetensors'
    ae: str = 'ae.sft'


async def caption_and_train(request: TrainRequest, run_id: str):
    """Handles captioning and training in the background."""
    dataset_dir = os.path.join(DATASETS_DIR, request.output_name)
    output_dir = os.path.join(OUTPUTS_DIR, request.output_name)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, f"{request.output_name}.toml")
    toml_content = f"""cache_latents = true
pretrained_model_name_or_path = "{MODELS_DIR}/{request.pretrained_model}"
ae = "{MODELS_DIR}/{request.ae}"
clip_l = "{MODELS_DIR}/{request.clip_l}"
t5xxl = "{MODELS_DIR}/{request.t5xxl}"
logging_dir = "{output_dir}/logs"
output_dir = "{output_dir}"
train_data_dir = "{dataset_dir}"
resolution = "{request.resolution},{request.resolution}"
network_alpha = {request.network_dim}
network_dim = {request.network_dim}
max_train_steps = {request.steps}
full_bf16 = {str(request.full_bf16).lower()}
unet_lr = {request.learning_rate}
mixed_precision = "bf16"
enable_bucket = true
output_name = "amante"
num_repeats = 1
epoch = 100
save_model_as = "safetensors"
save_precision = "fp16"
apply_t5_attn_mask = true
bucket_no_upscale = true
bucket_reso_steps = 64
cache_latents_to_disk = true
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
caption_extension = ".txt"
discrete_flow_shift = 3.0
dynamo_backend = "no"
gradient_accumulation_steps = 1
gradient_checkpointing = true
guidance_scale = 1.0
huber_c = 0.1
huber_scale = 1
huber_schedule = "snr"
loss_type = "l2"
lr_scheduler = "cosine"
lr_scheduler_args = []
lr_scheduler_num_cycles = 1
lr_scheduler_power = 1
max_bucket_reso = 2048
max_data_loader_n_workers = 0
max_grad_norm = 1
max_timestep = 1000
min_bucket_reso = 256
model_prediction_type = "raw"
network_args = [ "train_double_block_indices=all", "train_single_block_indices=all",]
network_module = "networks.lora_flux"
network_train_unet_only = true
noise_offset = 0.05
noise_offset_type = "Original"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False", "weight_decay=0.01",]
optimizer_type = "Adafactor"
prior_loss_weight = 1
sdpa = true
t5xxl_max_token_length = 512
text_encoder_lr = []
timestep_sampling = "flux_shift"
train_batch_size = 1
"""

    # Build training command
    command = f"""accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision bf16 --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 sd-scripts/flux_train_network.p --config_file {config_path}"""
    log_file = os.path.join(LOGS_DIR, f"{run_id}_train.log")

    try:
        # create log file and write to it
        with open(log_file, "w") as f:
            f.write(f"Training started for {request.output_name}\n")

        with open(config_path, "w") as f:
            f.write(toml_content)

        if len(request.image_urls):
            print("Downloading images")
            await asyncio.to_thread(download_images, DownloadRequest(output_name=request.output_name, urls=request.image_urls, captions=request.captions))

        if request.auto_captioning:
            print("Captioning images")
            await asyncio.to_thread(caption_images_in_directory, dataset_dir=dataset_dir)

        print("Running command:")
        print(command)
        
        with open(log_file, "w") as f:
            subprocess.Popen(command, shell=True, stdout=f, stderr=f, text=True)
    except Exception as e:
        print(f"Error starting training: {str(e)}")

@app.post("/run")
async def train_lora(request: TrainRequest, background_tasks: BackgroundTasks):
    """Starts the training asynchronously and returns the response immediately."""
    run_id = str(uuid.uuid4())
    background_tasks.add_task(caption_and_train, request, run_id)

    return {"message": "Training started", "run_id": run_id}

@app.get("/status/{run_id}")
def get_training_status(run_id: str):
    log_file = os.path.join(LOGS_DIR, f"{run_id}_train.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            progress = 0
            logs = f.readlines()[-10:]      
            is_completed = any("steps: 100%" in line.lower() for line in logs) if isinstance(logs, list) else False
            is_failed = any("returned non-zero exit status" in line.lower() for line in logs) if isinstance(logs, list) else False

            if is_failed:
                  return {"run_id": run_id, "status": "failed", "data": {"error": f"Error running job, check log file {run_id}_train.log"}}
            if is_completed:
                return {"run_id": run_id, "status": "completed", "data": {"progress": progress}}
            else:
                if isinstance(logs, list):
                    for line in logs:
                        if "steps:" in line.lower():
                            i = line.lower().find("%")
                            progress = int(line.lower()[i-4:i])
                return {"run_id": run_id, "status": "processing", "data": {"progress": progress}}
    return {"run_id": run_id, "status": "failed", "data": {"error": "Job not found"}}


@app.get("/logs/{run_id}")
def get_training_logs(run_id: str):
    log_file = os.path.join(LOGS_DIR, f"{run_id}_train.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = f.readlines()[-10:]      
            return {"run_id": run_id, "logs": logs}
    return {"run_id": run_id, "logs": "Job not found"}
