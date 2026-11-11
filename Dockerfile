FROM python:3.11-slim

WORKDIR /app
COPY requirements-infer.txt .
RUN pip install --no-cache-dir -r requirements-infer.txt

COPY infer.py .
COPY models/ models/
COPY samples/ samples/

ENTRYPOINT ["python", "infer.py"]
CMD ["--image", "samples/sample.jpg"]