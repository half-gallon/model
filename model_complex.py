import json
from math import ceil
import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

from constant import max_pad_len, n_mfcc


# -------------------------------------------
# --------- Hyperparameters? ---------------------
threshold_loss = 0.4  # train again if loss > threshold_loss

num_epochs = 300

# -------------------------------------------
# --------- Data Directory Configuration -----
DATA_DIR = "data"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")


# -------------------------------------------
# --------- Preprocessing -------------------
def extract_mfcc(file_path, max_pad_len=max_pad_len, n_mfcc=n_mfcc):
    """Extract MFCC from .wav file."""
    y, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width < 0:
        print(f"mfcc length is longer than max: {max_pad_len} < {mfcc.shape[1]}")
        raise 1
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    return mfcc


def load_data_from_directory(data_dir, max_pad_len=max_pad_len, n_mfcc=13):
    """Load .wav files from a directory and extract MFCC features."""
    X = []
    y = []

    # find sub directory
    sub_dirs = [
        sub_dir
        for sub_dir in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, sub_dir))
    ]

    print("sub_dirs", sub_dirs)

    for label, sub_dir in enumerate(
        [
            sub_dir
            for sub_dir in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, sub_dir))
        ]
    ):
        sub_dir_path = os.path.join(data_dir, sub_dir)

        if not os.path.isdir(sub_dir_path):  # Skip if not a directory
            continue

        for file_name in [f for f in os.listdir(sub_dir_path) if f.endswith(".wav")]:
            file_path = os.path.join(sub_dir_path, file_name)
            mfcc = extract_mfcc(file_path, max_pad_len, n_mfcc)
            X.append(mfcc)
            y.append([(1 if x == sub_dir else 0) for x in sub_dirs])

    return np.array(X), np.array(y)


# -------------------------------------------
# --------- LSTM Model Definition -----------
def random_indice(size):
    return [random.choice([j for j in range(size) if j != i]) for i in range(size)]


class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        batch_size,
        output_dim=2,  # number of classes
        num_layers=2,
        dropout=0.1,
    ):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # # Define the LSTM layer
        # self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # # Define the output layer
        # self.linear = nn.Linear(self.hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=4)
        self.relu2 = nn.ReLU()

        self.d1 = nn.Linear(in_features=8838, out_features=32)
        self.relu3 = nn.ReLU()

        self.d2 = nn.Linear(in_features=32, out_features=10)
        self.relu4 = nn.ReLU()

        self.d3 = nn.Linear(in_features=10, out_features=2)
        self.relu5 = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """
        x.shape = [batch, sequence, features] = [1, 190, 13]
        y.shape = [2] ==> [percent of 0 , percent of 1]

        return: Sum of prob of all classes
        """

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        # x = self.mp2(x)
        # print("after mp2", x.shape)

        x = x.flatten(start_dim=1)
        # print("after flatten", x.shape)

        x = self.d1(x)
        # print("after d1", x.shape)
        x = self.relu3(x)

        x = x.flatten(start_dim=1)
        x = self.d2(x)
        x = self.relu4(x)

        x = self.d3(x)
        x = self.relu5(x)

        # print("before softmax", x.shape)
        x = self.softmax(x)
        # print("after softmax", x.shape)

        return (x * y) > 0.5


# -------------------------------------------
# ------- Training & Evaluation --------------
def train_and_evaluate(
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=num_epochs
):
    """Train and evaluate the LSTM model."""
    input_dim = n_mfcc  # This corresponds to n_mfcc
    hidden_dim = max_pad_len  # Corresponding to max_pad_len
    batch_size = 1  # We're using individual samples
    output_dim = len(torch.unique(y_train_tensor))  # number of classes
    num_layers = 4

    print("output_dim", output_dim)

    model = Model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_loss_list = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # To accumulate loss for the entire epoch
        model.train()

        # Looping through individual samples
        for idx in random_indice(len(X_train_tensor)):
            model.zero_grad()

            x = X_train_tensor[idx].unsqueeze(0)
            y = y_train_tensor[idx].unsqueeze(0)

            y_pred = model(x, y)
            loss = torch.ones(1, requires_grad=True) - y_pred.long().sum()

            # print("\n")
            # print("y", y.reshape(-1).tolist())
            # print("y_pred", y_pred.reshape(-1).tolist())
            # print("loss", loss.item())

            is_owner = y.reshape(-1).tolist()[0] == 1
            if is_owner:
                loss = loss * 1.2  # false-positive is much important

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss_list.append(epoch_loss / len(X_train_tensor))
        # Print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Training Loss: {epoch_loss / len(X_train_tensor)}")

    final_loss = epoch_loss / len(X_train_tensor)
    print(f"Training finished with loss: {final_loss}")
    if final_loss > threshold_loss:
        print("Model is not trained well. train again")
        return train_and_evaluate(
            X_train_tensor,
            y_train_tensor,
            X_test_tensor,
            y_test_tensor,
            num_epochs=ceil(num_epochs * 1.2),
        )

    # Evaluation
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for idx in range(len(X_test_tensor)):
            x = X_test_tensor[idx].unsqueeze(0)
            y = y_test_tensor[idx].unsqueeze(0)

            y_val_pred = model(x, y)
            loss = torch.ones(1, requires_grad=True) - y_val_pred.long().sum()
            all_predictions.append(loss.item())

        correct_pred = (torch.tensor(all_predictions).sum()).float()
        acc = 1 - correct_pred.sum() / len(all_predictions)

    print(f"Validation Accuracy: {acc.item()}")

    real_input_x = X_train_tensor[0].unsqueeze(0)
    real_input_y = y_train_tensor[0].unsqueeze(0)
    torch_output = model(real_input_x, real_input_y)

    torch.onnx.export(
        model,
        (real_input_x, real_input_y),
        "network.onnx",
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=15,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    d = (real_input_x.detach().numpy()).reshape([-1]).tolist()
    dy = (real_input_y.detach().numpy()).reshape([-1]).tolist()

    data = dict(
        input_shapes=[
            list(real_input_x.size()),
            list(real_input_y.size()),
        ],
        input_data=[d, dy],
        output_data=[torch_output.detach().reshape([-1]).tolist()],
    )

    # Serialize data into file:
    json.dump(data, open("input.json", "w"), indent=2)


def main():
    # Load data
    X_train, y_train = load_data_from_directory(TRAIN_DATA_DIR)
    X_test, y_test = load_data_from_directory(TEST_DATA_DIR)

    # Convert to tensors

    # Changing shape to [batch, sequence, features]
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Changing shape to [batch, sequence, features]
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Train and evaluate
    train_and_evaluate(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)


if __name__ == "__main__":
    main()
