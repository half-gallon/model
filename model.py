import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from constants import device


# Define the GRU-based neural network model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
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

        # return out
        # Initialize hidden state
        if h0 is None:
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

    # Train the model
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            # Get the input data from the batch
            inputs = batch.squeeze(1).transpose(1, 2)

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

    initial_h0 = torch.zeros(num_layers, 1, hidden_size)
    dummy_input = (torch.randn(1, 190, 13), initial_h0)

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},  # Only the batch size is variable
            "output": {0: "batch_size"},
            "h0": {0: "num_layers", 1: "batch_size"},
        },
        verbose=True,
        input_names=["input", "h0"],
        output_names=["output"],
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
