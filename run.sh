#!/usr/bin/env bash
set -eo pipefail

# Create mount directory for service.
mkdir -p $FS_PATH

# mount file store if fs ip is set
if [ -n "$FS_SHARE" ]; then
  echo "Mounting Cloud Filestore."
  # if Cloud storage type is S3
  if [ "$CLOUD_STORAGE_TYPE" = "S3" ]; then
    mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport $FS_SHARE $FS_PATH
  fi

  # if Cloud storage type is GCS
  if [ "$CLOUD_STORAGE_TYPE" = "GCS" ]; then
    mount -o nolock -v $FS_SHARE $FS_PATH
  fi

  echo "Mounting completed."
fi

# Start the application
uvicorn main:app --host 0.0.0.0 --port $PORT &

# Exit immediately when one of the background processes terminate.
wait -n