FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1
WORKDIR /app
# Dependencies are installed from pyproject.toml alone, against an empty package, so the large
# layers below are rebuilt only when pyproject.toml changes, not on every source edit.
COPY pyproject.toml ./
RUN mkdir -p src/kolnote && touch src/kolnote/__init__.py README.md

# Channel bot only: no model, no GPU libraries. Pair it with an `openai_compat` STT engine.
FROM base AS channel
RUN pip install --no-cache-dir ".[http]"
COPY README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
ENTRYPOINT ["kolnote"]

# Speech-to-text server (`kolnote serve`): holds the model, needs a GPU or plenty of CPU.
FROM base AS server
ENV HF_HOME=/models \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
RUN pip install --no-cache-dir ".[faster-whisper]" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
COPY README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
ENTRYPOINT ["kolnote"]
CMD ["serve"]

# Default (`docker build .`): everything in one image, a bot with a local model.
FROM base AS full
ENV HF_HOME=/models \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
RUN pip install --no-cache-dir ".[faster-whisper,http,wyoming]" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
COPY README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
ENTRYPOINT ["kolnote"]
