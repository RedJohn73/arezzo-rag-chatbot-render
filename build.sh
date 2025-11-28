#!/usr/bin/env bash
# Force Python 3.10 on Render
echo "Python 3.10.13" > .python-version
pip install -r requirements.txt
