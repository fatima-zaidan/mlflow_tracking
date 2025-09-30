export DAGSHUB_USER="fatimazaidan"
export DAGSHUB_TOKEN="xxx"


docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="https://dagshub.com/fatima-zaidan/accelerator-tracking-demo.mlflow" \
  -e MLFLOW_TRACKING_USERNAME="$DAGSHUB_USER" \
  -e MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN" \
  -e DAGSHUB_OWNER="larawehbe" \
  -e DAGSHUB_REPO="accelerator-tracking-demo" \
  tracking-demos:latest