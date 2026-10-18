# FIASuite engine on Linux, headless and CPU only: every analysis tool, the selftest and PDF reports.
#   docker build -t fiasuite .
#   docker run --rm -v /path/to/models:/data/models:ro fiasuite
# FIASUITE_HOME=/data holds models/, config.json and fingerprints/ outside the image; ExifTool comes from apt.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libimage-exiftool-perl \
        libgl1 libegl1 libfontconfig1 libxkbcommon0 libdbus-1-3 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-linux.txt .
RUN pip install --no-cache-dir -r requirements-linux.txt
COPY fiasuite/ fiasuite/
COPY models/models.json models/models.json
ENV QT_QPA_PLATFORM=offscreen FIASUITE_HOME=/data
CMD ["sh", "-c", "python -m fiasuite --selftest fiasuite/guide_examples/ela_splice.jpg /tmp/selftest.json && cat /tmp/selftest.json"]
