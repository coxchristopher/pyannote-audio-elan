#!/bin/bash
#
# Set a number of environmental variables and locale-related settings needed
# for this recognizer to run as expected before calling the recognizer itself.
#
# It seems that recognizer processes invoked by ELAN don't inherit any regular
# environmental variables (like PATH), which makes it difficult to track down
# where both Python and ffmpeg(1) might be.  These same processes also have
# their locale set to C.  This implies a default ASCII file encoding, which
# causes some scripts to refuse to run (since many assume a more Unicode-
# friendly view of the world somewhere in their code).

# **
# ** Edit the following line to point to the directory in which 'ffmpeg' is
# ** found on this computer.
# **
export FFMPEG_DIR="/usr/bin/"

export LC_ALL="en_US.UTF-8"
export PYTHONIOENCODING="utf-8"
export PATH="$PATH:$FFMPEG_DIR"

# We need to set this environment variable to keep pyannote.audio from failing
# when run on MPS, since some operators (e.g., aten::_fft_r2c) haven't been
# implemented for MPS yet and need to fall back to running on a CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate the virtual environment, then execute the main recognizer script.
source ./venv-pyannote-audio-elan/bin/activate
exec python3 ./pyannote-audio-elan.py $1 >> ./elan_wrapper_debug.log 2>&1
