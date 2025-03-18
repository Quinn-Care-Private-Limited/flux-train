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
    return "OK"
class DownloadRequest(BaseModel):
    urls: list[str]
    output_name: str

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
    save_every_n_epochs: int = 0
    enable_bucket: bool = True
    full_bf16: bool = False


async def caption_and_train(request: TrainRequest, run_id: str, output_dir: str):
    """Handles captioning and training in the background."""
    os.makedirs(output_dir, exist_ok=True)

    # Build training command
    command = f"""accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 sd-scripts/flux_train_network.py \
    --pretrained_model_name_or_path {MODELS_DIR}/{request.pretrained_model} --clip_l {MODELS_DIR}/{request.clip_l} --t5xxl {MODELS_DIR}/{request.t5xxl} \
    --ae {MODELS_DIR}/{request.ae} --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
    --max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
    --network_module networks.lora_flux --network_dim {request.network_dim} --network_train_unet_only \
    --optimizer_type adamw8bit --learning_rate {request.learning_rate} \
    --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \
    --highvram --max_train_epochs {request.max_train_epochs} --save_every_n_epochs {request.save_every_n_epochs} --dataset_config {DATASETS_DIR}/{request.output_name}.toml \
    --output_dir {output_dir} --output_name {request.output_name} \
    --timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 --loss_type l2 \
    {"--enable_bucket" if request.enable_bucket else ""} {"--full_bf16" if request.full_bf16 else ""}
    """

    log_file = os.path.join(LOGS_DIR, f"{run_id}_train.log")

    try:
        if len(request.image_urls):
            print("Downloading images")
            await asyncio.to_thread(download_images, DownloadRequest(output_name=request.output_name, urls=request.image_urls))
        # Run dataset creation & captioning asynchronously
        print("Creating dataset config file")
        await asyncio.to_thread(create_dataset_config, DatasetConfig(output_name=request.output_name, trigger_word=request.trigger_word, num_repeats=request.num_repeats, resolution=request.resolution))

        print("Captioning images")
        await asyncio.to_thread(caption_images, CaptionRequest(output_name =request.output_name, trigger_word=request.trigger_word))

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
    output_dir = os.path.join(OUTPUTS_DIR, request.output_name)

    background_tasks.add_task(caption_and_train, request, run_id, output_dir)

    return {"message": "Training started", "run_id": run_id}

@app.get("/status/{run_id}")
def get_training_status(run_id: str):
    log_file = os.path.join(LOGS_DIR, f"{run_id}_train.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            progress = 0
            logs = f.readlines()[-10:]      
            is_completed = any("steps: 100%" in line.lower() for line in logs) if isinstance(logs, list) else False
            is_failed = any("exit status 1" in line.lower() for line in logs) if isinstance(logs, list) else False

            if is_failed:
                  return {"run_id": run_id, "status": "error", "data": "Error running job, check log file"}
            if is_completed:
                progress = 100
            else:
                 if isinstance(logs, list):
                    for line in logs:
                        if "steps:" in line.lower():
                            i = line.lower().find("%")
                            progress = int(line.lower()[i-3:i])

            return {"run_id": run_id, "status": "processing", "data": {"progress": progress}}
    return {"run_id": run_id, "status": "error", "data": "Job not found"}



