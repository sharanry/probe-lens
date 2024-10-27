import torch
import torch.nn as nn

from probe_lens.probes import LinearProbe


def test_linear_probe():
    input_dim = 10
    output_dim = 1
    batch_size = 5

    # Create a random input tensor
    x = torch.randn(batch_size, input_dim)

    # Initialize the linear probe
    model = LinearProbe(input_dim, output_dim)

    # Forward pass
    output = model(x)

    # Check the output shape
    assert output.shape == (
        batch_size,
        output_dim,
    ), f"Expected output shape {(batch_size, output_dim)}, but got {output.shape}"


def test_linear_probe_training():
    input_dim = 10
    output_dim = 1
    batch_size = 5
    x = torch.randn(batch_size, input_dim)
    A = torch.randn(input_dim, output_dim)
    B = torch.randn(output_dim)
    y = torch.sign(x @ A + B)

    model = LinearProbe(input_dim, output_dim)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True
    )
    model.train_probe(
        dataloader,
        dataloader,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=1000,
        verbose=True,
    )
    print("Model weights: ", model.linear.weight)
    print("Model bias: ", model.linear.bias)
    print("Model loss: ", nn.MSELoss()(model(x), y))
    assert nn.MSELoss()(model(x), y) < 1e-5
