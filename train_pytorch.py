import torch
from torch import nn

import argparse
from tqdm import tqdm
import pytorch_models
import dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch-size", default=512, type=int, help="Training batch size")
    parser.add_argument("--workers", default=1, type=int, help="Number of dataloader workers, may work incorrectly")
    parser.add_argument("--model_weights", default=None, type=str, help="The path to the file from which to load weights")
    parser.add_argument("--model_name", type=str, help="Name of the model architecture that will be trained")
    parser.add_argument("--data_path", type=str, help="Location of hdf5 containing the train data")

    args = parser.parse_args()

    dataloader = dataloader.DataGeneratorHDF5Pytorch(args.data_path)

    model: nn.Module = None

    if args.model_name == "resnet50-pretrained":
        model_config = {
            "base_model_name": "resnet50",
            "pretrained": False,
            "num_classes": 264
        }

        melspectrogram_parameters = {
            "n_mels": 128,
            "fmin": 20,
            "fmax": 16000
        }

        model = pytorch_models.ResNet()

        weights_path = args.model_weights

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)

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