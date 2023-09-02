import os
from pydub import AudioSegment


# -------------------------------------------
# --------- Parameters ---------------------
max_pad_len = 190
n_mfcc = 13

# -------------------------------------------
# --------- Data Directory Configuration -----
DATA_DIR = "raw-data"

"""Load .wav files from a directory and extract MFCC features."""

m4a_files = [
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".m4a")
]

for m4a_file in m4a_files:
    wav_file = m4a_file.replace(".m4a", ".wav")
    print(f"Transcode {m4a_file} to {wav_file}")

    sound = AudioSegment.from_file(m4a_file, format="m4a")
    file_handle = sound.export(wav_file, format="wav")
    os.remove(m4a_file)
