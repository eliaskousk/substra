import random

import pickle
import numpy as np
import torch
from torch import nn
import torch.utils.data as data

import substratools as tools


# TODO: Use CLI argument for these
# MODEL = "LinearRegression"
# MODEL = "CNN_OriginalFedAvg"
MODEL = "CNN_DropOut"
ONLY_DIGITS = True # For MNIST set to true - For FEMNIST set to either but only  false works for now
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
EPOCHS = 3


class LogisticRegression(torch.nn.Module):
    def __init__(self, only_digits=True):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(28 * 28, 10 if only_digits else 62)

    def forward(self, x):
        x = torch.squeeze(x)
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 28, 28))
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x


class CNN_DropOut(torch.nn.Module):
    """
    Recommended model by "Adaptive Federated Optimization" (https://arxiv.org/pdf/2003.00295.pdf)
    Used for EMNIST experiments.
    When `only_digits=True`, the summary of returned model is
    ```
    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
    _________________________________________________________________
    dropout (Dropout)            (None, 12, 12, 64)        0
    _________________________________________________________________
    flatten (Flatten)            (None, 9216)              0
    _________________________________________________________________
    dense (Dense)                (None, 128)               1179776
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                1290
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 28, 28))
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x


class AlgoLocal(tools.algo.Algo):

    def __init__(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True

        self.optimizer = "sgd"
        self.lr = 0.03
        self.wd = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, X, y, models, rank):

        data = self.create_dataloader(X, y)
        model = self.create_model()
        model.load_state_dict(models[0]) if models and models[0] else None
        model.to(self.device)
        model.train() # Set model to training mode

        # train and update
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr = self.lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr = self.lr,
                weight_decay = self.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(EPOCHS):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(data):
                x, labels = x.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        return len(X), model.cpu().state_dict()

    def predict(self, X, model):

        model_state = model
        model = self.create_model()
        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval() # Set model to evaluation mode

        with torch.no_grad():
            # X = torch.stack(X)
            # X = torch.squeeze(X, 1)
            X = X.to(self.device)
            pred = model(X)
            # pred_class = pred.cpu().numpy().argmax(1)
            _, pred_class = torch.max(pred, -1)

        return pred_class.cpu()

    def load_model(self, path):
        with open(path, 'rb') as f:
            # Discard the first field of the tuple (sample_num)
            _, model_state = pickle.load(f)
        return model_state

    def save_model(self, model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @classmethod
    def create_model(cls):
        if MODEL == "LinearRegression":
            return LogisticRegression(ONLY_DIGITS)
        elif MODEL == "CNN_DropOut":
            return CNN_DropOut(ONLY_DIGITS)
        elif MODEL == "CNN_OriginalFedAvg":
            return CNN_OriginalFedAvg(ONLY_DIGITS)
        else:
            return LogisticRegression(ONLY_DIGITS)

    @classmethod
    def create_dataloader(cls, X , y):
        y = torch.squeeze(y)
        return data.DataLoader(dataset = data.TensorDataset(X, y),
                               batch_size = TRAIN_BATCH_SIZE,
                               shuffle = True,
                               drop_last = False)

if __name__ == '__main__':
    tools.algo.execute(AlgoLocal())
