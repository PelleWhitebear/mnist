# main.py
import click
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import MyAwesomeModel
from data import mnist

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of training epochs")
@click.option("--model-path", default="trained_model.pth", help="path to save the trained model")
def train(lr, epochs, model_path):
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning Rate: {lr}, Epochs: {epochs}, Model Path: {model_path}")

    # Load MNIST data
    train_set, _ = mnist()

    # Initialize the model, optimizer, and criterion
    model = MyAwesomeModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy

    # Training loop
    for epoch in range(epochs):
        model.train()
        for data, target in train_set:
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model trained and saved at {model_path}")

    # Evaluate the model after training
    evaluate(model_path)

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(f"Model Checkpoint: {model_checkpoint}")

    # Load MNIST data
    _, test_set = mnist()

    # Initialize the model and load the trained weights
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_set:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test set: {100 * accuracy:.2f}%")

    # Check if accuracy meets the desired threshold
    if accuracy >= 0.85:
        print("Congratulations! Your model achieved at least 85% accuracy.")
    else:
        print("Your model did not achieve the desired accuracy. Please consider training for more epochs or adjusting hyperparameters.")

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
