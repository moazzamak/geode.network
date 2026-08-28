FROM rayproject/ray:2.56.1-py312-cpu

WORKDIR /workspace
COPY infrastructure/e7/requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt
COPY src ./src
COPY experiments ./experiments

ENV PYTHONPATH=/workspace