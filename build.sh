#!/usr/bin/env bash
set -o errexit

echo ">>> Forcing correct Gradio version…"
pip install --upgrade pip
pip uninstall -y gradio || true
pip install gradio==4.19.2

echo ">>> Installing all dependencies…"
pip install -r requirements.txt

echo ">>> Build completed."
