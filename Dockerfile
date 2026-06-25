FROM python:3.11-slim

# System dependencies for Camelot (Ghostscript) and pdf2image (Poppler),
# plus OpenCV runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "pdf_table_extractor.api:app", "--host", "0.0.0.0", "--port", "8000"]
