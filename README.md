# Flux Train - A Image Dataset and Training API

## Installation

### Pull docker image

```bash
docker pull quinninc/flux-train:1.0.0
```

### Create a `.env` file

```bash
nano .env
```

### Add environment variables

- PORT - Port to server
- MOUNT_PATH - Volume mount path (optional, dev only)
- FS_PATH - NFS filestore share url (optional)
- CLOUD_STORAGE_TYPE - GCS or S3 (optional)

### Run docker compose

```bash
docker compose up
```

OR

```bash
docker compose up -f docker-compose.dev.yml
```

## Overview

This API provides a comprehensive set of endpoints for downloading images, generating captions, creating dataset configurations, and training machine learning models, specifically designed for image-based machine learning workflows.

## Endpoints

### 1. Download Images

`POST /download-images`

**Request Model:** `DownloadRequest`

- `urls`: List of image URLs to download
- `output_name`: Name of the output directory for saved images

**Features:**

- Creates a directory for the dataset
- Downloads images from provided URLs
- Handles different image formats
- Converts images to JPEG
- Provides detailed download status for each URL

**Example Request:**

```python
{
    "urls": ["https://example.com/image1.png", "https://example.com/image2.jpg"],
    "output_name": "my_dataset"
}
```

**Response:**

```json
{
  "message": "Download complete",
  "files": {
    "https://example.com/image1.png": "/path/to/datasets/my_dataset/image1.jpg",
    "https://example.com/image2.jpg": "/path/to/datasets/my_dataset/image2.jpg"
  }
}
```

### 2. Caption Images

`POST /caption-images`

**Request Model:** `CaptionRequest`

- `output_name`: Directory containing images to caption
- `trigger_word`: Keyword to prepend to captions
- `prompt`: Optional prompt for image description (default: "Describe this image in detail.")

**Features:**

- Uses Florence-2 model for image captioning
- Supports CUDA acceleration
- Generates and saves captions as text files
- Handles various image formats

**Example Request:**

```python
{
    "output_name": "my_dataset",
    "trigger_word": "photo",
    "prompt": "Describe the main subject of this image"
}
```

**Response:**

```json
{
  "message": "Captions generated and saved",
  "captions": {
    "image1.jpg": "photo, a landscape with mountains and a lake",
    "image2.jpg": "photo, a portrait of a person smiling"
  }
}
```

### 3. Create Dataset Configuration

`POST /create-dataset-config`

**Request Model:** `DatasetConfig`

- `output_name`: Name of the dataset
- `trigger_word`: Classification token
- `num_repeats`: Number of times to repeat dataset (default: 10)
- `resolution`: Image resolution (default: 1024)
- `batch_size`: Training batch size (default: 1)
- More configuration options available

**Features:**

- Generates a TOML configuration file for machine learning training
- Configurable dataset parameters
- Supports custom training settings

**Example Request:**

```python
{
    "output_name": "my_dataset",
    "trigger_word": "landscape",
    "num_repeats": 5,
    "resolution": 512
}
```

**Response:**

```json
{
  "message": "Dataset config created",
  "path": "/path/to/datasets/my_dataset.toml"
}
```

### 4. Train Model

`POST /train`

**Request Model:** `TrainRequest`

- `output_name`: Name for the trained model
- `max_train_epochs`: No of epochs to train (optional)
- `learning_rate`: Learning rate of the training (optional)
- `network_dim`: Network dimension of lora (optional)
- `save_every_n_epochs`: Save models at N epochs (optional)
- `pretrained_model`: Relative path of flux model to NFS_SHARE (optional)
- `clip_l`: Relative path of Clip large model to NFS_SHARE (optional)
- `t5xxl`: Relative path of T5xxl model to NFS_SHARE (optional)
- `ae`: Relative path of ae model to NFS_SHARE (optional)
- `enable_bucket`: Enable or Disable image bucket for multi aspect ratio dataset (optional)
- `full_bf16`: Enable or Disable full bf16 training (optional)

**Features:**

- Launches training using `accelerate`
- Supports various model configurations
- Generates unique run ID for tracking
- Writes training logs

**Example Request:**

```python
{
    "output_name": "my_model",
    "max_train_epochs": 5,
    "learning_rate": 8e-4
    "pretrained_model": 'flux1-dev.sft'
    "clip_l": 'clip_l.safetensors'
    "t5xxl": 't5xxl_fp16.safetensors'
    "ae": 'ae.sft'
    "network_dim": 4
    "save_every_n_epochs": 1
    "enable_bucket": true
    "full_bf16": false
}
```

**Response:**

```json
{
  "message": "Training started",
  "run_id": "unique-run-identifier"
}
```

### 5. Training Status

`GET /status/{run_id}`

**Features:**

- Retrieve status of a specific training run
- Returns last 10 log lines

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PIL
- Requests
- Accelerate

## Error Handling

- Comprehensive error tracking for each endpoint
- HTTP status codes for different error scenarios
- Detailed error messages in responses

## Performance Considerations

- CUDA acceleration support
- Configurable batch sizes
- Mixed precision training
- Gradient checkpointing

## Security

- Timeout for image downloads
- File extension validation
- Isolated dataset and model directories

## Logging

- Unique run IDs
- Detailed log files for each training session
- Configurable logging levels

## Contributions

- Follow PEP 8 guidelines
- Write comprehensive docstrings
- Include type hints
- Add unit tests for new functionality

## License

MIT

## Acknowledgments

- Florence-2 Model
- Hugging Face Transformers
- PyTorch
- Accelerate

```

This documentation provides a comprehensive overview of the API, covering all endpoints, request/response structures, features, configuration options, and best practices. The documentation is formatted in GitHub-flavored Markdown and includes detailed explanations and example requests/responses.

Would you like me to elaborate on any specific section or modify the documentation?
```
