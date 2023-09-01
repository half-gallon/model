import os

from read_dir import read_dir
from model import train_model

import os
import torchaudio

import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from constants import max_time_dim

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

max_time_dim = 190

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchaudio
import torchaudio.transforms as T


def collate_fn(batch):
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

    dataset = collate_fn(dataset)
    print("After padding")
    for x in dataset:
        print(x.shape)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Define the GRU-based neural network model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # # # Reshape the input tensor
        # x = x.squeeze(
        #     1
        # )  # Remove the channel dimension. Now the shape is [batch_size, 13, 190].
        # x = x.transpose(
        #     1, 2
        # )  # Swap the last two dimensions to get shape [batch_size, 190, 13].

        # # Initialize hidden state
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # # GRU forward pass
        # out, _ = self.gru(x, h0)

        # # Take the output from the last time step and pass it through a linear layer
        # out = self.fc(out[:, -1, :])

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU forward pass
        out, _ = self.gru(x, h0)

        # Take the output from the last time step and pass it through a linear layer
        out = self.fc(out[:, -1, :])

        return out


def train_model(
    dataloader,
    input_size=13,  # n_mfcc
    hidden_size=64,
    output_size=1,
    num_layers=3,
    learning_rate=0.001,
    num_epochs=20,
):
    """
    Train the model with the given dataloader.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader object with training data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (nn.Module): The trained model.
    """
    model = GRUModel(input_size, hidden_size, output_size, num_layers)

    # Set the model to training mode
    model.train()

    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean squared error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dummy_input = torch.randn(1, 190, 13)

    # Train the model
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            # Get the input data from the batch
            inputs = batch.squeeze(1).transpose(1, 2)
            dummy_input = inputs

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Assuming your dataset provides labels, you'd compute the loss like this:
            # loss = criterion(outputs, labels)
            # For the sake of this example, I'll use a dummy tensor for labels:
            # labels = torch.Tensor(np.array([1 for _ in range(len(inputs))]))
            labels = torch.Tensor(np.array([1 for _ in range(len(inputs))])).unsqueeze(
                1
            )

            print("len(labels)", len(labels))

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
    print("Training finished!")

    model.eval()
    model.to("cpu")

    # dummy_input = torch.randn(1, 190, 13)

    torch.onnx.export(
        model,
        dummy_input,
        # (dummy_input),
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        # dynamic_axes={
        #     "input": {0: "batch_size"},  # Only the batch size is variable
        #     "output": {0: "batch_size"},
        # },
        verbose=True,
        # input_names=["input"],
        # output_names=["output"],
    )

    # torch.onnx.export(
    #     model,  # model being run
    #     torch.randn((1, 13, 190)),
    #     "model.onnx",  # where to save the model (can be a file or file-like object)
    #     export_params=True,  # store the trained parameter weights inside the model file
    #     opset_version=11,  # the ONNX version to export the model to
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     dynamic_axes={
    #         "input": {0: "batch_size", 1: "sequence_length", 2: "feature_dim"},
    #         "output": {0: "batch_size"},
    #     },
    #     verbose=True,  # store the trained parameter weights inside the model file
    #     input_names=["input"],  # specify the name of the inputs
    #     output_names=["output"],  # specify the name of the outputs
    # )

    return model


DATA_DIR = "data"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")


os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)

########################################################################
# Load files
########################################################################


TRAIN_DATA_FILES = read_dir(TRAIN_DATA_DIR)
TEST_DATA_FILES = read_dir(TEST_DATA_DIR)

print(
    f"Training data - {len(TRAIN_DATA_FILES)}",
)
print(
    f"Test data     -  {len(TEST_DATA_FILES)}",
)


########################################################################
# Load data
########################################################################


dataloader = load_training_data(TRAIN_DATA_DIR, batch_size=len(TEST_DATA_FILES))
model = train_model(dataloader, num_epochs=2)
