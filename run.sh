#!/usr/bin/env bash
# One-command launcher for macOS / Linux
set -e
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
