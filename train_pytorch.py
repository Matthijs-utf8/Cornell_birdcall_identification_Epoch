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
    parser.add_argument("--model-weights", default=None, type=str, help="The path to the file from which to load weights")
    parser.add_argument("--model-name", type=str, help="Name of the model architecture that will be trained")
    parser.add_argument("--data-path", type=str, help="Location of hdf5 containing the train data")

    args = parser.parse_args()


    model = None

    if args.model_name == "resnet50-pretrained":
        model = pytorch_models.ResNet()# .to('cuda:0')

        weights_path = args.model_weights

        if weights_path != None:
            model.load_state_dict(
                torch.load(weights_path)["model_state_dict"]
            )

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    print_loss_frequency = 5
    time_since_print_loss = 0

    total_loss = 0.0

    for i in range(10 ** 9):
        pass
    print("done looping")

    with dataloader.DataGeneratorHDF5Pytorch(args.data_path) as dataloader:
        for epoch in range(args.epochs):
            print("epoch =", epoch)
            for data in dataloader:
            # for data in tqdm(dataloader):
                inputs, labels = data
                # labels = labels.to("cuda:0")
                # inputs = inputs.to("cuda:0")

                optimizer.zero_grad()

                outputs = model(inputs)
                
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                time_since_print_loss += 1

                if time_since_print_loss % print_loss_frequency == print_loss_frequency - 1:
                    print("loss =", total_loss / print_loss_frequency)
                    total_loss = 0.0

print("done")