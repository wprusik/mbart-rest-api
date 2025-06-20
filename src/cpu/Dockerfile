FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn transformers torch tiktoken protobuf sentencepiece

COPY app.py /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
