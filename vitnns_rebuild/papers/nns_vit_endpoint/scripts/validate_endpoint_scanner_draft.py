#!/usr/bin/env python3
"""Run the validator for the current portable publication."""
from pathlib import Path
import importlib.util
import os
import runpy
import sys
repo=Path(__file__).resolve().parents[3]
if importlib.util.find_spec("numpy") is None:
    interpreter=repo.parent/".conda/envs/vitnns/bin/python"
    os.execv(str(interpreter),[str(interpreter),str(Path(__file__).resolve())])
runpy.run_path(str(Path(__file__).with_name("validate_publication_r2.py")),run_name="__main__")
