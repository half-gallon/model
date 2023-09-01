import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

# --------- Data Directory Configuration -----
DATA_DIR = "data"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")


# --------- Preprocessing -------------------
def extract_mfcc(file_path, max_pad_len=190, n_mfcc=13):
    """Extract MFCC from .wav file."""
    y, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    return mfcc


def load_data_from_directory(data_dir, max_pad_len=190, n_mfcc=13):
    """Load .wav files from a directory and extract MFCC features."""
    X = []
    y = []

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
            y.append(label)

    return np.array(X), np.array(y)


# -------------------------------------------
# --------- LSTM Model Definition -----------
class LSTMClassifier(nn.Module):
    """LSTM Classifier model."""

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.softmax(
            self.linear(lstm_out[:, -1, :])
        )  # Extract the output from the last time step
        return y_pred


# -------------------------------------------
# ------- Training & Evaluation --------------
def train_and_evaluate(
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=10
):
    """Train and evaluate the LSTM model."""
    input_dim = 13  # This corresponds to n_mfcc
    hidden_dim = 190  # Corresponding to max_pad_len
    batch_size = 1  # We're using individual samples now
    output_dim = len(torch.unique(y_train_tensor))  # number of classes
    num_layers = 2

    print("output_dim", output_dim)

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    model.train()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # To accumulate loss for the entire epoch
        model.train()

        # Looping through individual samples
        for idx in range(len(X_train_tensor)):
            model.zero_grad()
            y_pred = model(
                X_train_tensor[idx].unsqueeze(0)
            )  # adding an extra batch dimension
            loss = loss_function(
                y_pred, y_train_tensor[idx].unsqueeze(0)
            )  # adding an extra batch dimension
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Training Loss: {epoch_loss / len(X_train_tensor)}")

    # Evaluation
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for idx in range(len(X_test_tensor)):
            y_val_pred = model(X_test_tensor[idx].unsqueeze(0))
            _, y_pred_tags = torch.max(y_val_pred, dim=1)
            all_predictions.append(y_pred_tags.item())

        correct_pred = (torch.tensor(all_predictions) == y_test_tensor).float()
        acc = correct_pred.sum() / len(correct_pred)

    print(f"Validation Accuracy: {acc.item()}")

    real_input = X_train_tensor[0].unsqueeze(
        0
    )  # Take the first element and add an extra batch dimension

    print("real_input", real_input)
    print("real_input.shape", real_input.shape)

    torch.onnx.export(
        model,
        real_input,
        "lstm-3.onnx",
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


# Load data
X_train, y_train = load_data_from_directory(TRAIN_DATA_DIR)
X_test, y_test = load_data_from_directory(TEST_DATA_DIR)

print("y_train", y_train)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(
    0, 2, 1
)  # Changing shape to [batch, sequence, features]
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(
    0, 2, 1
)  # Changing shape to [batch, sequence, features]
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Train and evaluate
train_and_evaluate(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
