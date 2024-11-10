#!/bin/bash



GDRIVE_CLIENT_ID=54417335472-d7c6rouorers3rfcvtt2b273pr449gp7.apps.googleusercontent.com
GDRIVE_CLIENT_SECRET=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

dvc remote modify --local gdrive gdrive_client_id "$GDRIVE_CLIENT_ID"
dvc remote modify --local gdrive gdrive_client_secret "$GDRIVE_CLIENT_SECRET"
dvc pull
uv sync
uv run coverage run -m pytest tests/
uv run coverage xml
docker build -t dogbreedclassifier-hydraconfigs:dogbreed-classifier-withhydraconfigs .
docker run --rm \
  -v ${PWD}/logs:/app/logs \
  dogbreedclassifier-hydraconfigs:dogbreed-classifier-withhydraconfigs \
  python src/train.py experiment=dogbreed_ex

