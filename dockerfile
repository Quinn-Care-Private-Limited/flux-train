FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

### Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive \
  ### Prefer binary wheels over source distributions for faster pip installations
  PIP_PREFER_BINARY=1 \
  ### Ensures output from python is printed immediately to the terminal without buffering
  PYTHONUNBUFFERED=1 

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git wget && \
  pip install --upgrade pip

### Clean up to reduce image size
RUN apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/* 

WORKDIR /app
# ADD files
ADD main.py requirements.txt ./
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git
RUN cd sd-scripts/ && pip install -r requirements.txt && cd .. && pip install -r requirements.txt
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

ENV PORT=80

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", $PORT]