import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the GRU-based neural network model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape the input tensor
        x = x.squeeze(
            1
        )  # Remove the channel dimension. Now the shape is [batch_size, 13, 181].
        x = x.transpose(
            1, 2
        )  # Swap the last two dimensions to get shape [batch_size, 181, 13].

        # Initialize hidden state
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
            inputs = batch
            print("batch size:", len(batch))

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Assuming your dataset provides labels, you'd compute the loss like this:
            # loss = criterion(outputs, labels)
            # For the sake of this example, I'll use a dummy tensor for labels:
            labels = torch.Tensor(np.array([1 for _ in range(len(inputs))]))

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )
    print("Training finished!")

    # export model to onnx file
    model.eval()
    torch.onnx.export(
        model,  # model being run
        torch.randn((1, 13, 181)),
        "model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        # dynamic_axes={
        #     "input": {0: "batch_size"},  # variable length axes
        #     "output": {0: "batch_size"},
        # },
        verbose=True,  # store the trained parameter weights inside the model file
        input_names=["input"],  # specify the name of the inputs
        output_names=["output"],  # specify the name of the outputs
    )

    return model
