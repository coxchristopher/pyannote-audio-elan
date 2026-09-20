#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A short script that wraps the speaker diarization services provided by
# pyannote.audio (https://github.com/pyannote/pyannote-audio) to act as
# a local recognizer in ELAN.

#
# TODO:
#
#   * Reimplement the VAD module as its own recognizer (since it's no
#     longer compatible with the 4.x.x releases of pyannote.audio; code
#     stripped out of here now, need to resurrect from 3.x.x recognizer)
#

import csv
import glob
import html
import os
import os.path
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import timeit

import cpuinfo
import numpy
import pyannote.audio
import pyannote.audio.pipelines
import pyannote.audio.pipelines.utils.hook
import pyannote.audio.telemetry
import scipy.spatial.distance
import torch

#DEFAULT_EMBEDDING_MODEL = 'speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e'
#DEFAULT_EMBEDDING_MODEL = 'speechbrain/spkrec-ecapa-voxceleb'
DEFAULT_EMBEDDING_MODEL = 'pyannote/wespeaker-voxceleb-resnet34-LM'

# A subclass of ProgressHook that provides updates on the status of a running
# speech service in the format that ELAN's recognizer API expects.
class ELANProgressHook(pyannote.audio.pipelines.utils.hook.ProgressHook):
    def __init__(self, transient = False):
        self.stage = 0
        # This is hard-coded to the current release of pyannote.audio, where
        # speaker diarization involves four stages of processing and voice
        # activity detection only one.
        self.num_stages = 4

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, step_name, step_artifact, file = None, total = None,
        completed = None):
        if completed is None:
            completed = total = 1

        if not hasattr(self, "step_name") or step_name != self.step_name:
            self.step_name = step_name
            self.stage = self.stage + 1

        # Rather than count from 0-100% completed for each stage, count from
        # 0-25% for the first of four stages, 25-50% for the second of four
        # stages, etc.
        progress = (min(1.0, (completed / total)) / self.num_stages) + \
            ((self.stage - 1) / self.num_stages)

        # When reporting progress percentages to the user, ELAN checks to see
        # if an output (XML or timeseries) file has been created once progress
        # is reported to be at or over 100%.  If we haven't created an output
        # file by then, ELAN displays a warning dialog and/or a prompt for the
        # user to create new tiers.  By scaling back the progress by a small
        # fraction of a percent here, we can avoid those warnings and make
        # sure that the user is only prompted to create new tiers once (i.e.,
        # once we output "DONE" below).
        progress = max(0, progress - 0.01)
        step = step_name.capitalize()

        print(f"PROGRESS: {progress:.2f} {step}, {completed} of {total}",
            flush = True)

# The parameters provided by the user via the ELAN recognizer interface
# (specified in CMDI).
params = {}

# Parameters for the pipeline.
pipeline_params = {}


# Disable pyannote.audio telemetry for the current session.
pyannote.audio.telemetry.set_telemetry_metrics(False)

# Read in all of the parameters that ELAN passes to this local recognizer on
# stdin.
for line in sys.stdin:
    match = re.search(r'<param name="(.*?)".*?>(.*?)</param>', line)
    if match:
        params[match.group(1)] = match.group(2).strip()

if not params.get('output_segments', ''):
    print("ERROR: missing output parameter!", flush = True)
    sys.exit(-1)

# File names passed in via the ELAN recognizer may have certain characters
# XML/HTML-escaped (e.g., "&apos;" for "'", etc.).  Turn those back into
# their non-escaped equivalents before using these as references to actual
# files below.
params['output_segments'] = html.unescape(params['output_segments'])
params['source'] = html.unescape(params['source'])
params['use_checkpoint'] = html.unescape(params['use_checkpoint'])
params['checkpoint'] = html.unescape(params['checkpoint'])

# Read in the Hugging Face authentication token, either from the user's
# provided CMDI parameters (overriding any default token that is available)
# or from the default Hugging Face token file:
#
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
token = params.get('auth_token', '')
if token:
    token = html.unescape(token)
else:
    token_path = os.path.join(\
        os.path.expanduser('~'), '.cache', 'huggingface', 'token')
    if os.path.isfile(token_path):
        with open(token_path, 'rt') as token_file:
            token = token_file.readline()
assert token, "ERROR: missing Hugging Face authentication token"

# Prepare to perform speaker verification, if requested.
speaker_embedding_pipeline = None
speaker_id_to_embedding = {}

# Read and set parameters for the pipeline.
mode_specific_args = {}

# Some CMDI parameters for speaker diarization relate to keyword arguments
# that are provided when applying the pipeline to a specific audio file,
# rather than (hyper-)parameters that are used to instantiate the pipeline.
# We gather these into a dictionary that is provided as keyword arguments
# when the pipeline is applied.
num_speakers = params['num_speakers']
if num_speakers != 'Unknown':
    mode_specific_args['num_speakers'] = int(num_speakers)

min_speakers = params['min_speakers']
if min_speakers != '_':
    mode_specific_args['min_speakers'] = int(min_speakers)

max_speakers = params['max_speakers']
if max_speakers != '_':
    mode_specific_args['max_speakers'] = int(min_speakers)

# If the user has provided a speaker verification configuration file (a CSV
# file with two columns, 'id' (speaker ID) and 'audio' (path to audio file
# containing speech sample for the individual represented by this speaker ID),
# parse that configuration file and generate embeddings for each speaker
# based on the provided audio.
speaker_verification_csv = params.get('speaker_verification_csv', '')
if speaker_verification_csv:
    speaker_embedding_pipeline = pyannote.audio.pipelines.\
        speaker_verification.PretrainedSpeakerEmbedding(\
            DEFAULT_EMBEDDING_MODEL, token = token)

    speaker_verification_dir = \
        os.path.dirname(os.path.abspath(speaker_verification_csv))
    with open(speaker_verification_csv, 'r', encoding = 'utf-8-sig') \
              as speaker_verification_file:
        speaker_ver_dict = csv.DictReader(speaker_verification_file) 
        for line in speaker_ver_dict:
            audio_fname = os.path.join(speaker_verification_dir, \
                os.path.basename(line['audio']))

            # Load and down-sample the audio to 16KHz as needed.
            audio = pyannote.audio.Audio(sample_rate = 16000)
            waveform, rate = audio(audio_fname)

            speaker_id_to_embedding[line['id']] = \
                speaker_embedding_pipeline(waveform[None])

# If we've been given a (valid) model checkpoint to use for segmentation, use
# it to instantiate the pipeline for the service that the user requested.
pipeline = None
if params['use_checkpoint'] == 'True':
    # If we've been asked to use a checkpoint, but the one that the user
    # provided isn't available, try to load the one supplied with pyannote-
    # audio-elan.
    checkpoint = params['checkpoint']
    if not os.path.isfile(checkpoint):
        checkpoints = glob.glob('*.ckpt')
        if not checkpoints:
            print("ERROR: Custom segmentation model requested, but none "\
                  "available", flush = True)
            sys.exit(-1)

        checkpoint = checkpoints[0]
        print(f"Found default segmentation model {checkpoint}", flush = True)

    print("Creating a diarization pipeline with the segmentation model", \
        flush = True)
    pipeline = pyannote.audio.pipelines.SpeakerDiarization(\
        segmentation = checkpoint, embedding = DEFAULT_EMBEDDING_MODEL)

    # Specify minimum duration off, the segmentation threshold, and the 
    # clustering threshold, the latter two having been finetuned as per:
    #
    #   https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/adapting_pretrained_pipeline.ipynb
    pipeline_params = pipeline.default_parameters()
    pipeline_params['segmentation']['min_duration_off'] = \
        float(params['min_duration_off'])
    # In pyannote.audio 'community-1', the segmentation threshold parameter
    # only exists for non-powerset models, which are no longer the default.
    if not pipeline._segmentation.model.specifications.powerset:
        pipeline_params['segmentation']['threshold'] = \
            float(params['segmentation_threshold'])
    pipeline_params['clustering']['threshold'] = \
        float(params['clustering_threshold'])

# Otherwise, use a pre-trained diarization pipeline from Hugging Face.
else:
    print("Loading the speaker diarization pipeline from Hugging Face...",
        flush = True)
    pipeline = pyannote.audio.Pipeline.from_pretrained(\
        "pyannote/speaker-diarization-community-1",
         token = token)

# Use the given parameters with this pipeline.
print("Apply parameters to pipeline...", flush = True)
pipeline = pipeline.instantiate(pipeline_params)

# Send the pipeline to an accelerator (when possible).
print("Loaded pipeline, sending to accelerator if possible...", flush = True)
device = 'cpu'
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # For now, we only try to off-load processing onto an MPS back-end on
    # M-series Apple processors, not on Intel ones, since pyannote.audio
    # doesn't ever (appear to) finish processing on an MPS device using a
    # discrete GPU on Intel Macs.
    if 'apple m' in cpuinfo.get_cpu_info().get('brand_raw').lower():
        device = 'mps'
        pipeline.to(torch.device('mps'))
elif torch.cuda.is_available() and torch.backends.cuda.is_built():
    device = 'cuda'
    pipeline.to(torch.device('cuda'))

# Perform the requested service on the given audio.
print("Applying pipeline to audio...", flush = True)
output = None
with ELANProgressHook() as hook:
    start = timeit.default_timer()
    output = pipeline(params["source"], hook = hook, **mode_specific_args)
    end = timeit.default_timer()
    print(f"Applying pipeline on {device} took {end - start}s", flush = True)

# Gather up the speech segments identified for each speaker by the pipeline.
speakers = {}
for turn, speaker in output.speaker_diarization:
    if not speaker in speakers:
        speakers[speaker] = []
    speakers[speaker] = speakers[speaker] + [(turn.start, turn.end)]

# If we've been asked to, attempt to verify which embedding returned by the
# diarization pipeline matches up with which speaker (among those for whom
# audio samples and speaker IDs were provided in the speaker verification
# config file).
if speaker_embedding_pipeline:
    identified_speakers = {}
    for s, diarization_speaker_id in enumerate(speakers.keys()):
        # Convert the one-dimensional array returned by the speaker verifi-
        # cation pipeline into a two-dimensional one with the shape (1, n)
        # that our cosine distance measure below expects.
        diarization_embedding = numpy.reshape(\
            output.speaker_embeddings[s], (1, -1))

        min_distance = 0.0
        best_matching_speaker_id = None
        for (ref_speaker_id, ref_embedding) in speaker_id_to_embedding.items():
            dist = scipy.spatial.distance.cdist(diarization_embedding,
                ref_embedding, metric = "cosine")[0, 0]
            print(f"Comparing {diarization_speaker_id} with "\
                  f"{ref_speaker_id} = {dist} (current min. "\
                  f"distance = {min_distance})", flush = True)
            if dist > min_distance:
                min_distance = dist
                best_matching_speaker_id = ref_speaker_id

        if best_matching_speaker_id:
            print(f"Speaker {diarization_speaker_id} is "\
                  f"{best_matching_speaker_id}")
            identified_speakers[best_matching_speaker_id] = \
                speakers[diarization_speaker_id]

    speakers = identified_speakers

# Open 'output_segments' for writing, and return all of the segments of speech
# recognized by pyannote.audio as the contents of <span> elements.
with open(params['output_segments'], 'w', encoding = 'utf-8') as output_segs:
    # Write document header.
    output_segs.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output_segs.write('<TIERS xmlns:xsi="http://www.w3.org/2001/XMLSchema-'\
        'instance" xsi:noNamespaceSchemaLocation="file:avatech-tiers.xsd">\n')

    for speaker in speakers:
        if speaker_embedding_pipeline:
            output_segs.write(f'<TIER columns="{speaker}">\n')
        else:
            output_segs.write(f'<TIER columns="PyannoteAudio_{speaker}">\n')

        # Write out annotations (e.g., '<span start="17.492" end="18.492">
        # <v></v></span>').
        for (start, end) in speakers[speaker]:
            output_segs.write('    '\
                f'<span start="{start:.3f}" end="{end:.3f}"><v></v></span>\n')

        output_segs.write('</TIER>\n')

    output_segs.write('</TIERS>\n')

# Finally, tell ELAN that we're done.
print('RESULT: DONE.', flush = True)
