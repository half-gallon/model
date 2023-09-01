import os
import random
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import json

from load_training_data import load_training_data, load_training_data_array
from read_dir import read_dir


DATA_DIR = "data"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)


TRAIN_DATA_FILES = read_dir(TRAIN_DATA_DIR)
TEST_DATA_FILES = read_dir(TEST_DATA_DIR)

train_data = load_training_data_array(TRAIN_DATA_FILES)
first_input = train_data[0]

print("batch size: ", len(train_data))

print("first input shape: ", first_input.shape)

######################################################################


n_class = 2
n_hidden = 13  # 은닉층 사이즈


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, dropout=0.3)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(torch.float))
        self.b = nn.Parameter(torch.randn([n_class]).type(torch.float))
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden_and_cell):
        x = x.transpose(0, 1)
        outputs, hidden = self.lstm(x, hidden_and_cell)
        outputs = outputs[-1]  # 최종 예측 Hidden Layer
        model = torch.mm(outputs, self.W) + self.b  # 최종 예측 최종 출력 층
        return model


# model = nn.LSTM(190, 13, 3)  # Input dim is 3,3 output dim is 3
model = Model()  # Input dim is 3,3 output dim is 3

model.train()
# criterion = nn.MSELoss()  # Mean squared error
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Flips the neural net into inference mode
model.eval()
model.to("cpu")

for inputs in train_data:
    batch_size = 1
    hidden = torch.zeros(3, 1, 13, requires_grad=True)
    cell = torch.zeros(3, 1, 13, requires_grad=True)

    outputs = model(inputs, (hidden, cell))

    print("len(inputs)", len(inputs))
    print("inputs.shape", inputs.shape)
    print("outputs", outputs)
    print("len(outputs)", len(outputs))
    print("outputs[0].shape", outputs[0].shape)

    # labels = torch.Tensor(np.array([1 for _ in range(len(inputs))]))  # .unsqueeze(1)
    labels = torch.ones(1, len(inputs), requires_grad=True)

    print("labels", labels)
    print("labels.shape", labels.shape)

    loss = criterion(outputs, labels.squeeze(dim=-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Export the model
torch.onnx.export(
    model,  # model being run
    # model input (or a tuple for multiple inputs)
    first_input,
    # where to save the model (can be a file or file-like object)
    "lstm-model-2.onnx",
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

data_array = ((first_input).detach().numpy()).reshape([-1]).tolist()

data_json = dict(input_data=[data_array])

print(data_json)

# Serialize data into file:
json.dump(data_json, open("lstm-input-2.json", "w"))
