# pyannote-audio-elan

pyannote-audio-elan provides access to voice activity detection and overlap-aware speaker diarization services provided by [pyannote.audio](https://github.com/pyannote/pyannote-audio) ([Plaquet & Bredin 2023](https://www.isca-archive.org/interspeech_2023/plaquet23_interspeech.html), [Bredin 2023](https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html)) from directly inside [ELAN](https://tla.mpi.nl/tools/tla-tools/elan/).  This allows users to apply both out-of-the-box and fine-tuned segmentation/diarization models to multimedia sources linked to ELAN transcripts from directly within ELAN's user interface.

In addition to performing voice activity detection and speaker diarization, pyannote-audio-elan can optionally apply speaker verification to the results of these segmentation processes, attempting to determine the most likely match between a known set of speakers' voices and the speaker(s) identified during automatic segmentation.  When given a set of speaker identifiers (e.g., "CDC", "BRS", etc.) and short audio samples (e.g., 30 seconds of speech) from each corresponding individual, pyannote-audio-elan will return tiers from the segmentation process with names that contain the speaker identifier (e.g., returning a tier named "CDC", rather than "PyannoteAudio\_Speaker\_00", if the audio sample provided for the speaker "CDC" is the closest match to the speaker Speaker\_00 identified in automatic segmentation).

## Requirements and installation

pyannote-audio-elan makes use of several of other open-source applications and utilities:

* [ELAN](https://tla.mpi.nl/tools/tla-tools/elan/) (tested with ELAN 6.7-6.9
  under macOS 13-15)
* [Python 3](https://www.python.org/) (tested with Python 3.10)
* [ffmpeg](https://ffmpeg.org)

pyannote-audio-elan is written in Python 3, and also depends on a number of Python packages that can be installed using `pip` in a virtual environment. Under macOS 15, the following commands can be used to fetch and install the necessary Python packages:
```
git clone https://github.com/coxchristopher/pyannote-audio-elan
cd pyannote-audio-elan

python3.10 -m venv venv-pyannote-audio-elan
source venv-pyannote-audio-elan/bin/activate
pip install -r requirements.txt
chmod +x pyannote-audio-elan.sh
```

Once all of these tools and packages have been installed, pyannote-audio-elan can be made available to ELAN as follows:

1. Edit the file `pyannote-audio-elan.sh` to specify (a) the directory in which ffmpeg is located, and (b) a Unicode-friendly language and locale (if `en_US.UTF-8` isn't available on your computer).
2. To make pyannote-audio-elan available to ELAN, move your pyannote-audio-elan directory into ELAN's `extensions` directory.  This directory is found in different places under different operating systems:
   
   * Under macOS, right-click on `ELAN_6.9` in your `/Applications`
     folder and select "Show Package Contents", then copy your
     `pyannote-audio-elan` folder into `ELAN_6.9.app/Contents/app/extensions`.
   * Under Linux, copy your `pyannote-audio-elan` folder into
     `ELAN_6-9/app/extensions`.
   * Under Windows, copy your `pyannote-audio-elan` folder into
     `C:\Users\AppData\Local\ELAN_6-9\app\extensions`.

Once ELAN is restarted, it will now include two new options in the list of services found under the 'Recognizer' tab in Annotation Mode: 'pyannote.audio speaker diarization with speaker verification' (for overlap-aware speaker diarization with optional speaker verification) and 'pyannote.audio voice activity detection' (for basic voice activity detection, without any speaker diarization applied).  The user interfaces for both recognizers allow users to enter the settings needed to apply speaker diarization media linked to this ELAN transcript (e.g., optionally specifying the exact number of speakers that are present in this recording, if known, or a maximum number of speakers that may be present, which can improve the accuracy of speaker diarization).

Once these settings have been entered in pyannote-audio-elan, pressing the `Start` button will begin applying the selected segmentation service to the media.  Once that process is complete, if no errors occurred, ELAN will allow the user to load the resulting tier(s) with the automatically recognized segments into the current transcript.

Importantly, pyannote-audio-elan currently requires access to pyannote.audio's [segmentation](https://huggingface.co/pyannote/segmentation-3.0) and [speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) pipelines on [Hugging Face](https://huggingface.co).  Both of these pipelines require users to read and accept their conditions on Hugging Face before using them, and to provide a Hugging Face access token to download them the first time they are used.  While we hope to be able to offer a version of pyannote-audio-elan in the future that removes this requirement and works entirely offline, for now, users of pyannote-audio-elan will need to:

1. Accept the [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) user conditions,
2. Accept the [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) user conditions, then
3. Create an access token at [https://hf.co/settings/tokens](https://hf.co/settings/tokens) that can be copied into the pyannote-audio-elan settings.

## Limitations

This is an alpha release of pyannote-audio-elan, and has only been tested under macOS (13-15) with Python 3.10.  No support for Windows or Linux is included in this version.

As noted above, installing and using pyannote-audio-elan currently requires an internet connection (at least the first time that pyannote-audio-elan is used, so that the segmentation and diarization pipelines can be downloaded from Hugging Face) and some familiarity with command-line software development tools.  We hope to reduce (and, ideally, eliminate) these requirements in the future, providing pre-packaged, offline-friendly versions of these recognizers that offer more user-friendly installation options (see the pyannote.audio
[tutorial](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb) on offline speaker diarization and Lorena Martín Rodríguez's [SileroVAD-Elan](https://github.com/l12maro/SileroVAD-Elan) project for examples of how this might be done).

## Acknowledgements

Thanks are due to the developers of [pyannote.audio](https://github.com/pyannote/pyannote-audio) for the pipelines that this recognizer relies upon and the accompanying documentation,
particularly of the fine-tuning process.  Thanks, as well, to [Han Sloetjes](https://www.mpi.nl/people/sloetjes-han) for his help with issues related to ELAN's local recognizer specifications.

## Citing pyannote-audio-elan

If referring to this code in a publication, please consider using the following citation:

> Cox, Christopher. 2025. pyannote-audio-elan: An implementation of pyannote.audio speaker diarization and voice activity detection services as a recognizer for ELAN. Version 0.1.0.

```
@manual{cox25pyannoteaudioelan,
    title = {pyannote-audio-elan: An implementation of pyannote.audio speaker diarization and voice activity detection services as a recognizer for {ELAN}.},
    author = {Christopher Cox},
    year = {2025}
    note = {Version 0.1.0},
    }
```
