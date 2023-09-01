# print(TRAIN_DATA_FILES)
# print(TEST_DATA_FILES)


########################################################################
# Preprocess data
########################################################################


import torchaudio
import torchaudio.transforms as T


def preprocess_wav(
    filename, desired_sample_rate=16000, n_mfcc=13, n_fft=400, hop_length=160
):
    """
    Preprocess a given audio file (.wav).

    Parameters:
        filename (str): Path to the audio file (.wav).
        desired_sample_rate (int): Desired sample rate after resampling.
        n_mfcc (int): Number of MFCCs to compute.
        n_fft (int): FFT window size.
        hop_length (int): Stride or hop length.

    Returns:
        Tensor: Preprocessed audio tensor.
    """

    # Load the .wav file
    waveform, sample_rate = torchaudio.load(filename)

    # Resample the waveform if needed
    if sample_rate != desired_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=desired_sample_rate)
        waveform = resampler(waveform)

    # Extract MFCC features
    mfcc_transform = T.MFCC(
        sample_rate=desired_sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft, "hop_length": hop_length},
    )
    mfcc = mfcc_transform(waveform)

    # Normalize features (e.g., mean and standard deviation normalization)
    mfcc = (mfcc - mfcc.mean()) / mfcc.std()

    return mfcc
