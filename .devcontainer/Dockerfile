FROM ghcr.io/astral-sh/uv:0.8.17-python3.12-bookworm-slim

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        libmagic1 \
        pandoc \
        unixodbc \
    && rm -rf /var/lib/apt/lists/*
