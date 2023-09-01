import pathlib

from pydub import AudioSegment

import os


# convert .mp3 files to .wav files
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")


# read .mp3 or .wav files from directory
def read_dir(directory):
    files = []
    # convert .mp3 file to .wav file
    for filepath in pathlib.Path(directory).glob("*.mp3"):
        mp3_file_path = str(filepath.absolute())
        wav_file_path = mp3_file_path.replace(".mp3", ".wav")
        convert_mp3_to_wav(mp3_file_path, wav_file_path)
        os.remove(mp3_file_path)
        print(
            f".mp3 file is replaced with .wav file: {mp3_file_path} -> {wav_file_path}"
        )

    # get .wav files
    for filepath in pathlib.Path(directory).glob("*.wav"):
        files.append(str(filepath.absolute()))
    return files
