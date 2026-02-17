#!/bin/sh

python3 -m pip install pip setuptools wheel
python3 -m pip install -e ".[dev]"
python3 -m pip install -U ./llm_utils
