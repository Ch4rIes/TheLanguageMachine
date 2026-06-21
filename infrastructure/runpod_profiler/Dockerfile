FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

COPY infrastructure/runpod_profiler/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY core/language_machine ./language_machine
COPY infrastructure/runpod_profiler/handler.py ./handler.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
