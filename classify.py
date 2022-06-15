# Imports
import argparse
import glob
import os
import torch
import torch.nn as nn
from torch.optim import Adam

import models as Model
import dataset as Dataset

MODEL_NAMES = ['CNN', 'AlexNet', 'ResNet', 'MobileNet', 'VGG']


class CNNClassification():
    def __init__(self, args):
        self.model = Model.selectModel(args.model.lower(), args.n_class, args.pretrained)
        self.train_loader = Dataset.load_train(args.train_path, args.pretrained,
                                               args.batch_size)
        self.test_loader = Dataset.load_test(args.test_path, args.pretrained,
                                             args.batch_size)
        self.result_path = args.results_path
        self.num_epochs = args.epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.train_count = len(glob.glob(args.train_path + '/**/*.png'))
        self.test_count = len(glob.glob(args.test_path + '/**/*.png'))

    def check_path(self):
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

    def training(self):
        # Set model in training mode
        self.model.train()
        total_step = self.train_count
        # Iterate over epochs
        for epoch in range(self.num_epochs):

            # Iterate the dataset
            for i, (images, labels) in enumerate(self.train_loader):
                # Get batch of samples and labels
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.
                          format(epoch + 1, self.num_epochs, i + self.batch_size, total_step, loss.item()))

        self.check_path()
        # Save the model checkpoint
        torch.save(self.model.state_dict(), self.result_path + '/model.ckpt')

    def testing(self):
        # Load the model
        self.model.load_state_dict(torch.load(self.result_path + '/model.ckpt'))

        # Test the model
        self.model.eval()  # Set the model in evaluation mode
        n_test_imgs = self.test_count

        # Compute testing accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                # get network predictions
                outputs = self.model(images)

                # get predicted class
                _, predicted = torch.max(outputs.data, 1)

                # compare with the ground-truth
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the {} test images: {} %'.format(n_test_imgs, 100 * correct / total))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-path', default='data/train', dest='train_path', type=str,
                        help="Path containing a folder with your training data")
    parser.add_argument('--test-path', default='data/test', dest='test_path', type=str,
                        help="Path containing a folder with your testing data")
    parser.add_argument('--result-path', default='results', dest='results_path', type=str,
                        help="Path containing a folder with your testing data")
    parser.add_argument('--lr', default=.001, dest='lr', type=float,
                        help="Learning rate for backpropagation")
    parser.add_argument('--batch-size', default=10, dest='batch_size', type=int,
                        help="Batch size")
    parser.add_argument('--model', default='CNN', type=str, dest='model', choices=MODEL_NAMES,
                        help='model architecture: ' +
                             ' | '.join(MODEL_NAMES) +
                             ' (default: CNN)')
    parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
                        help="Model pretrained")
    parser.add_argument('--epochs', default=5, dest='epochs', type=int,
                        help="Number of epochs")
    parser.add_argument('--num-class', default=3, dest='n_class', type=int,
                        help="Number of classes")
    return parser


def main(params):
    print("Running with {} model {}".format(params.model, '-pretrained' if params.pretrained else '-not pretrained'))
    classification = CNNClassification(params)
    classification.training()
    classification.testing()


if __name__ == '__main__':
    main(get_args().parse_args())
