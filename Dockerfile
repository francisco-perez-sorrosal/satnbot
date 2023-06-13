FROM python:3.11-slim-bullseye as base

# Create the working directory
#   set -x prints commands and set -e causes us to stop on errors
RUN set -ex && mkdir /bot
WORKDIR /bot

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

FROM base as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.4.0

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-root

# Copy only the relevant directories
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY . .

# Build image:
# docker build -t "satnbot" .

# Run image:
# docker run -it --rm -e DISCORD_TOKEN="MTA-xxxxx" -e OPENAI_API_KEY="sk-xxx" satnbot
ENTRYPOINT ["./entrypoint.sh"]