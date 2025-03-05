FROM nvidia/cuda:12.4.1-base-ubuntu22.04

USER root

### Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive \
  ### Prefer binary wheels over source distributions for faster pip installations
  PIP_PREFER_BINARY=1 \
  ### Ensures output from python is printed immediately to the terminal without buffering
  PYTHONUNBUFFERED=1 

ENV PORT=80
ENV FS_PATH=/mnt/fs

# Install dependencies
RUN apt-get update && apt-get install -y python3-dev python3-pip git wget tini nfs-common libtool build-essential && \
  pip install --upgrade pip

### Clean up to reduce image size
RUN apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/* 

WORKDIR /app
# ADD files
COPY requirements.txt ./
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git && cd sd-scripts/ && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY main.py run.sh ./
RUN chmod +x run.sh

# Start FastAPI server
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/run.sh"]