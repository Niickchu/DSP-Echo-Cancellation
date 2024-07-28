import os
import librosa
import soundfile
from datetime import datetime
import numpy as np

def get_current_folder():
    return os.path.dirname(os.path.abspath(__file__))


def read_audio(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return signal, sample_rate
    pass


def save_audio(file_path, data, sample_rate):
    soundfile.write(file_path, data, sample_rate)

def get_datetime_string():
    now = datetime.now()
    return now.strftime("%m-%d-%Y_%H-%M-%S")