FROM python:3.10-slim  # يُفضل استخدام 3.10 لكونه أكثر استقراراً مع DeepFace


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]