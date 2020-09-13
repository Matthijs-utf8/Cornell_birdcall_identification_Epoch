import torch
from torch import nn

import argparse
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=512, type=int, help="Training batch size")
    parser.add_argument("--workers", default=1, type=int, help="Number of dataloader workers, may work incorrectly")
    parser.add_argument("--model_weights", default=None, type=str, help="The path to the file from which to load weights")
    parser.add_argument("--model_name", type=str, help="Name of the model architecture that will be trained")

    args = parser.parse_args()

    dataloader = ['test']

    model = None
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.params(), args.lr)

    print_loss_frequency = 2000
    time_since_print_loss = 0

    total_loss = 0.0


    for epoch in range(args.epochs):
        print("epoch =", epoch)
        for data in tqdm(dataloader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            time_since_print_loss += 1

            if time_since_print_loss % print_loss_frequency == print_loss_frequency - 1:
                print("loss =", total_loss / print_loss_frequency)
                total_loss = 0.0

print("done")