# GEODE runtime image — ROCm base (the M258 cell: written to spec;
# the container build is not executed in this environment).
#
# Build:  docker build -t geode-ml .
# Run:    docker run --device=/dev/kfd --device=/dev/dri -p 8000:8000 geode-ml
FROM rocm/pytorch:rocm6.3_ubuntu22.04_py3.11

# the cache root is baked in so every dispatch shell inherits the
# registered environment (the standalone-RX-9070 convention)
ENV GEODE_CACHE_DIR=/opt/geode/data/cache \
    HIP_VISIBLE_DEVICES=0

WORKDIR /opt/geode
COPY pyproject.toml README.md ./
COPY geode ./geode
RUN pip install --no-cache-dir '.[api]'

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD \
    python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)" \
    || exit 1

# the API is local-only by design; bind to the loopback inside the
# container and map the port explicitly at run time
CMD ["uvicorn", "geode.api.service:app", "--host", "127.0.0.1", "--port", "8000"]
