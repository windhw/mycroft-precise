# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import numpy as np
import os
import sys
from typing import *

from precise.params import pr, Vectorizer
from precise.util import load_audio, InvalidAudio
from sonopy import mfcc_spec, mel_spec
from precise.tflite_audio import tflite_mfccs

inhibit_t = 0.4
inhibit_dist_t = 1.0
inhibit_hop_t = 0.1

vectorizers = {
    Vectorizer.mels: lambda x: mel_spec(
        x, pr.sample_rate, (pr.window_samples, pr.hop_samples),
        num_filt=pr.n_filt, fft_size=pr.n_fft
    ),
    Vectorizer.mfccs: lambda x: mfcc_spec(
        x, pr.sample_rate, (pr.window_samples, pr.hop_samples),
        num_filt=pr.n_filt, fft_size=pr.n_fft, num_coeffs=pr.n_mfcc
    ),
    Vectorizer.speechpy_mfccs: lambda x: __import__('speechpy').feature.mfcc(
        x, pr.sample_rate, pr.window_t, pr.hop_t, pr.n_mfcc, pr.n_filt, pr.n_fft
    ),
    Vectorizer.tflite_mfccs: lambda x: tflite_mfccs(
        x, os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio.tflite")
    )
}


def vectorize_raw(audio: np.ndarray) -> np.ndarray:
    """Turns audio into feature vectors, without clipping for length"""
    if len(audio) == 0:
        raise InvalidAudio('Cannot vectorize empty audio!')
    return vectorizers[pr.vectorizer](audio)


def add_deltas(features: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(features)
    for i in range(1, len(features)):
        deltas[i] = features[i] - features[i - 1]

    return np.concatenate([features, deltas], -1)


def vectorize(audio: np.ndarray) -> np.ndarray:
    try:
       """
       Args:
           audio: Audio verified to be of `sample_rate`

       Returns:
           array<float>: Vector representation of audio
       """
       if len(audio) > pr.max_samples:
           audio = audio[-pr.max_samples:]
       features = vectorize_raw(audio)
       if len(features) < pr.n_features:
           features = np.concatenate([
               np.zeros((pr.n_features - len(features), features.shape[1])),
               features
           ])
       if len(features) > pr.n_features:
           features = features[-pr.n_features:]
    except Exception as e:
        print("[HW_DEBUG]:Exception is:" + str(e) )
        sys.exit(1)
    return features


def vectorize_delta(audio: np.ndarray) -> np.ndarray:
    return add_deltas(vectorize(audio))


def vectorize_inhibit(audio: np.ndarray) -> np.ndarray:
    """
    Returns an array of inputs generated from the
    wake word audio that shouldn't cause an activation
    """

    def samp(x):
        return int(pr.sample_rate * x)

    inputs = []
    for offset in range(samp(inhibit_t), samp(inhibit_dist_t), samp(inhibit_hop_t)):
        if len(audio) - offset < samp(pr.buffer_t / 2.):
            break
        inputs.append(vectorize(audio[:-offset]))
    return np.array(inputs) if inputs else np.empty((0, pr.n_features, pr.feature_size))
