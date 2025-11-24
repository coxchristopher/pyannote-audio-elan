#!/bin/bash
#
# Set a number of environmental variables and locale-related settings needed
# for this recognizer to run as expected before calling the recognizer itself.
#
# It seems that recognizer processes invoked by ELAN have their locale set
# to C.  This implies a default ASCII file encoding, which causes some
# scripts to refuse to run (since many assume a more Unicode-friendly view
# of the world somewhere in their code).

export LC_ALL="en_US.UTF-8"
export PYTHONIOENCODING="utf-8"

# We need to set this environment variable to keep pyannote.audio from failing
# when run on MPS, since some operators (e.g., aten::_fft_r2c) haven't been
# implemented for MPS yet and need to fall back to running on a CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate the virtual environment, then execute the main recognizer script.
source ./venv-pyannote-audio-elan/bin/activate
exec python3 ./pyannote-audio-elan.py $1 >> ./elan_wrapper_debug.log 2>&1
