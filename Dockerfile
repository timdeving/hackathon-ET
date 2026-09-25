# Submission image: the judges' two commands, run inside the container.
#   docker build -t team .
#   docker run --rm --gpus all --network none \
#       -v /data/test:/data/test:ro -v "$PWD/outputs:/app/outputs" team \
#       python run_submission.py --videos /data/test --out outputs/predictions.json
# PyTorch's CUDA wheels bring their own CUDA libraries; `--gpus all` supplies the driver.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.in requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "predictions.json"]
