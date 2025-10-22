#!/usr/bin/env bash
set -euo pipefail

# Load env vars from .env if present
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs -d '\n')
fi

# Sanity checks
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set. Edit .env and add your rotated key." >&2
  exit 1
fi
if [ -z "${GOOGLE_SHEET_ID:-}" ]; then
  echo "GOOGLE_SHEET_ID is not set. Edit .env with your spreadsheet id." >&2
  exit 1
fi
if [ -z "${GOOGLE_SERVICE_ACCOUNT_FILE:-}" ] && [ -z "${GOOGLE_SERVICE_ACCOUNT_JSON_BASE64:-}" ]; then
  echo "Provide GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON_BASE64 in .env." >&2
  exit 1
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
