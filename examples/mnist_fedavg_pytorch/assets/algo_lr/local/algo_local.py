import random

import pickle
import numpy as np
import torch
from torch import nn

import substratools as tools


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

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
        for epoch in range(3):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(zip(X, y)):
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
            X = torch.stack(X)
            X = torch.squeeze(X, 1)
            X = X.to(self.device)
            pred = model(X)
            # pred_class = pred.cpu().numpy().argmax(1)
            _, pred_class = torch.max(pred, -1)

        return pred_class

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
        return LogisticRegression(28 * 28, 10)


if __name__ == '__main__':
    tools.algo.execute(AlgoLocal())
