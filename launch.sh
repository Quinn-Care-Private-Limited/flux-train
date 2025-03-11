#!/usr/bin/env bash
gcloud compute instances create-with-container $1 \
  --project=ai-rnd-431419 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --accelerator=count=1,type=nvidia-l4 \
  --tags=http-server,https-server \
  --image=projects/cos-cloud/global/images/cos-stable-117-18613-164-49 \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-balanced \
  --boot-disk-device-name=$1 \
  --container-image=quinninc/flux-train:1.1.0 \
  --metadata-from-file user-data=cloud-config.yaml