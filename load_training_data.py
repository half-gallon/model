import os
import torchaudio

import torch
from torch.utils.data import Dataset

from preprocess_wav import preprocess_wav
from torch.utils.data import DataLoader

from constants import max_time_dim


def collate_fn_tensor(batch):
    # batch is a list of waveform tensors outputted by `__getitem__` for each file.
    # Since each waveform can be of a different length (last dimension),
    # we pad them to the maximum length in the batch.

    # Extract the time dimension (last dimension) for each sample

    # Pad waveform tensors to have the same size in the time dimension
    waveforms_padded = [
        torch.nn.functional.pad(item, (0, max_time_dim - item.shape[-1]))
        for item in batch
    ]

    # return torch.stack(waveforms_padded, dim=0)
    return waveforms_padded


def collate_fn_list(batch):
    for item in batch:
        print(item)

    waveforms_padded = [
        torch.nn.functional.pad(item, (0, max_time_dim - item.shape[-1]))
        for item in batch
    ]

    # return torch.stack(waveforms_padded, dim=0)
    return waveforms_padded


class AudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on an audio file.
        """
        self.directory = directory
        self.transform = transform

        # List all files in directory
        self.file_list = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) & f.endswith(".wav")
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_list[idx])
        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(file_path)

        return waveform


def load_training_data(directory, batch_size=1):
    """
    # Usage:
    dataloader = load_training_data()
    for batch in dataloader:
        print(
            batch.shape
        )  # This will print the shape of each batch of preprocessed audio files


    Load training data from 'data' directory.

    Args:
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader object with training data.
    """
    # Initialize the dataset and DataLoader
    dataset = AudioDataset(directory=directory, transform=preprocess_wav)

    print("Before padding")
    for x in dataset:
        print(x.shape)

    dataset = collate_fn_tensor(dataset)
    print("After padding")
    for x in dataset:
        print(x.shape)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_training_data_array(files):
    mfccs = []
    ret = []
    for file in files:
        mfcc = preprocess_wav(file)
        mfccs.append(mfcc)
        ret.append(torch.tensor(mfcc))

    return collate_fn_tensor(ret)
