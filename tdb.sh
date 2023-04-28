#!/bin/bash

set -ex

export DISCORD_TOKEN="<YOUR_DISCORD_APP_TOKEN_HERE>"
export OPENAI_API_KEY="<YOUR_OPENAI_TOKEN_HERE>"

# For pyllms
export LLMS_DEFAULT_MODEL="gpt-3.5-turbo"
export KAGI_API_KEY="<YOUR_KAGI_API_KEY_HERE>"

poetry run python satnbot.py