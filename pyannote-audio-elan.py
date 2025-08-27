#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A short script that wraps the voice activity detection and speaker
# diarization services provided by pyannote.audio (https://github.com/
# pyannote/pyannote-audio) to act as a local recognizer in ELAN.

import csv
import html
import os
import os.path
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import cpuinfo
import numpy
import pyannote.audio
import pyannote.audio.pipelines
import pyannote.audio.pipelines.utils.hook
import scipy.spatial.distance
import torch

#DEFAULT_EMBEDDING_MODEL = 'speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e'
DEFAULT_EMBEDDING_MODEL = 'speechbrain/spkrec-ecapa-voxceleb'

# A subclass of ProgressHook that provides updates on the status of a running
# speech service in the format that ELAN's recognizer API expects.
class ELANProgressHook(pyannote.audio.pipelines.utils.hook.ProgressHook):
    def __init__(self, transient = False, mode = 'Diarization'):
        self.stage = 0
        # This is hard-coded to the current release of pyannote.audio, where
        # speaker diarization involves four stages of processing and voice
        # activity detection only one.
        if mode == 'Diarization':
            self.num_stages = 4
        else:
            self.num_stages = 1

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
params['source'] = html.unescape(params['source'])
params['checkpoint'] = html.unescape(params['checkpoint'])

# Determine which mode this script is meant to run in (voice activity
# detection or speaker diarization), based on the first argument provided to
# this script.
params['mode'] = 'VAD' if (len(sys.argv) > 1 and sys.argv[1] == 'VAD') \
    else 'Diarization'

# Prepare to perform speaker verification, if requested.
speaker_embedding_pipeline = None
speaker_id_to_embedding = {}

# Read and set parameters for the pipeline.
mode_specific_args = {}
if params['mode'] == 'VAD':
    pipeline_params = {
        "onset": float(params['onset']),
        "offset": float(params['offset']),
        "min_duration_on": float(params['min_duration_on']),
        "min_duration_off": float(params['min_duration_off'])
    }
elif params['mode'] == 'Diarization':
    pipeline_params = {
        "segmentation": {
            "min_duration_off": float(params['min_duration_off'])
        }
    }

    # Some CMDI parameters for speaker diarization relate to keyword
    # arguments that are provided when applying the pipeline to a specific
    # audio file, rather than (hyper-)parameters that are used to instantiate
    # the pipeline.  We gather these into a dictionary that is provided as
    # keyword arguments when the pipeline is applied.
    num_speakers = params['num_speakers']
    if num_speakers != 'Unknown':
        mode_specific_args['num_speakers'] = int(num_speakers)

    min_speakers = params['min_speakers']
    if min_speakers != '_':
        mode_specific_args['min_speakers'] = int(min_speakers)

    max_speakers = params['max_speakers']
    if max_speakers != '_':
        mode_specific_args['max_speakers'] = int(min_speakers)

    # If the user has provided a speaker verification configuration file (a
    # CSV file with two columns, 'id' (speaker ID) and 'audio' (path to audio
    # file containing speech sample for the individual represented by this
    # speaker ID), parse that configuration file and generate embeddings for
    # each speaker based on the provided audio.
    speaker_verification_csv = params.get('speaker_verification_csv', '')
    if speaker_verification_csv:
        mode_specific_args['return_embeddings'] = True
        speaker_embedding_pipeline = pyannote.audio.pipelines.\
            speaker_verification.PretrainedSpeakerEmbedding(\
                DEFAULT_EMBEDDING_MODEL, use_auth_token = params["auth_token"])

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
if os.path.isfile(params['checkpoint']):
    print("Loading the checkpoint for the segmentation model...", flush = True)
    model = pyannote.audio.Model.from_pretrained(params['checkpoint'])

    if params['mode'] == 'VAD':
        print("Creating a VAD pipeline with the seg. model", flush = True)
        pipeline = pyannote.audio.pipelines.VoiceActivityDetection(\
            segmentation = model)

    elif params['mode'] == 'Diarization':
        pipeline = pyannote.audio.pipelines.SpeakerDiarization(\
            segmentation = model, 
            embedding = DEFAULT_EMBEDDING_MODEL,
#            embedding = "speechbrain/spkrec-ecapa-voxceleb",
            clustering = "AgglomerativeClustering")

        # Need to specify additional hyperparameters if we're using our own
        # checkpoint as part of this diarization pipeline.  The list of para-
        # meters required to instantiate this pipeline can be retrieved via
        # pipeline.parameters(), and the already-instantiated parameters via
        # pipeline.parameters(instantiated = True) (as defined in pyannote-
        # pipeline/pyannote/pipeline/pipeline.py).  Default values specified
        # here are taken directly from:
        #
        #    https://github.com/FrenchKrab/IS2023-powerset-diarization/
        #
        # with the exception of the segmentation and clustering thresholds,
        # which were fine-tuned as per:
        #
        #   https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/adapting_pretrained_pipeline.ipynb
#        pipeline_params['segmentation']['threshold'] = 0.4442333667381752
#        pipeline_params['segmentation']['threshold'] = 0.5577202404940658 # v0
        pipeline_params['segmentation']['threshold'] = 0.5577202404940658 # v1
        pipeline_params['clustering'] = {
            "method": "centroid",
            "min_cluster_size": 15,
#            "threshold": 0.7153814381597874,
#            "threshold": 0.6380173939509877,   # v0
            "threshold": 0.6939225490462559,    # v1
        }

# Otherwise, use a pre-trained segmentation model from Hugging Face.
else:
    if params['mode'] == 'VAD':
        print("Loading the VAD pipeline from Hugging Face...", flush = True)
        pipeline = pyannote.audio.Pipeline.from_pretrained(\
            "pyannote/voice-activity-detection",
             use_auth_token = params["auth_token"])

    elif params['mode'] == 'Diarization':
        print("Loading the speaker diarization pipeline from Hugging Face...",
               flush = True)
        pipeline = pyannote.audio.Pipeline.from_pretrained(\
            "pyannote/speaker-diarization-3.0",
             use_auth_token = params["auth_token"])

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
embeddings = []
with ELANProgressHook(mode = params['mode']) as hook:
    import timeit
    start = timeit.default_timer()
    if params['mode'] == 'Diarization' and speaker_embedding_pipeline:
        output, embeddings = pipeline(params["source"], hook = hook, 
            **mode_specific_args)
    else:
        output = pipeline(params["source"], hook = hook, **mode_specific_args)
    end = timeit.default_timer()
    print(f"DEBUG: Processing on {device} took {end - start}s")

# Gather up the speech segments identified for each speaker by the pipeline.
speakers = {}
for turn, _, speaker in output.itertracks(yield_label = True):
    if not speaker in speakers:
        speakers[speaker] = []
    speakers[speaker] = speakers[speaker] + [(turn.start, turn.end)]

# If we've been asked to, attempt to verify which embedding returned by the
# diarization pipeline matches up with which speaker (among those for whom
# audio samples and speaker IDs were provided in the speaker verification
# config file).
print(f"Have {len(embeddings)} embeddings, {len(output.labels())} output labels")
if speaker_embedding_pipeline:
    identified_speakers = {}
    for s, diarization_speaker_id in enumerate(output.labels()):
        # For whatever reason, the diarization pipeline returns one-dimensional
        # arrays with the shape (192,), rather than the two-dimensional ones 
        # with the shape (1, 192) that this speaker verification pipeline
        # returns (and that our cosine distance measure below expects).
        diarization_embedding = numpy.reshape(embeddings[s], (1, 192))

        min_distance = 1.0  # 0 = identical, 1 = opposite
        best_matching_speaker_id = None
        for (ref_speaker_id, ref_embedding) in speaker_id_to_embedding.items():
            dist = scipy.spatial.distance.cdist(diarization_embedding,
                ref_embedding, metric = "cosine")[0, 0]
            print(f"Comparing {diarization_speaker_id} with {ref_speaker_id} = {dist} (current min. distance = {min_distance})", flush = True)
            if dist < min_distance:
                min_distance = dist
                best_matching_speaker_id = ref_speaker_id

        if best_matching_speaker_id:
            print(f"Speaker {diarization_speaker_id} is {best_matching_speaker_id}")
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
