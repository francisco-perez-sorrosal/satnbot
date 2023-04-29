#!/bin/bash

set -ex

export DISCORD_TOKEN="<YOUR_DISCORD_APP_TOKEN_HERE>"
export OPENAI_API_KEY="<YOUR_OPENAI_TOKEN_HERE>"

# For pinecone
export PINECONE_API_KEY="<YOUR_PINECONE_API_KEY_HERE>"

poetry run python satnbot.py