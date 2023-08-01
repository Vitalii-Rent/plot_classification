import numpy as np
import torch
import sys
sys.path.append('src')
from data.nn_utils import BatchSamplerSimilarLength, create_embeddings
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm
import torch.nn as nn
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def validate_epoch(model, data, collate_batch, batch_size=100, subset_size=None):
  if subset_size:
    data, _ = random_split(data, [subset_size, len(data) - subset_size])

  dataloader = DataLoader(data,
                          batch_sampler=BatchSamplerSimilarLength(
                                            dataset=data,
                                            batch_size=100),
                          collate_fn=collate_batch)

  total_instances = total_correct =  total_loss = 0
  with torch.no_grad():
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y).item() * len(X)
        total_loss += loss
        pred = torch.argmax(pred, dim=1)
        correct_predictions = sum(pred == y).item()
        total_correct += correct_predictions
        total_instances += len(y)

  return total_loss/total_instances, total_correct/total_instances


def train(model,
          writer,
          train_subset,
          val_subset,
          collate_batch,
          n_epochs=30,
          batch_size=32,
          pretrained_epochs=0,
          verbose=True,
          check_early_stop=True,
          patience=3):
    # Hold the best model
    best_loss = np.inf   # init to infinity
    best_weights = None

    early_stopper = EarlyStopper(patience=patience)
    train_dataloader = DataLoader(train_subset,
                                  batch_sampler=BatchSamplerSimilarLength(
                                                    dataset=train_subset,
                                                    batch_size=batch_size),
                                  collate_fn=collate_batch)

    for epoch in range(pretrained_epochs, n_epochs + pretrained_epochs):
        model.train()
        loop = tqdm.tqdm(train_dataloader, unit="batch", mininterval=0, disable=not verbose)
        loop.set_description(f"Epoch {epoch}")

        for X_batch, y_batch in loop:

            X_batch = X_batch.to(device)
            y_batch = y_batch.squeeze().to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer = model.get_optimizer()
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            loop.set_postfix(cross_entropy=loss.item())
        # evaluate accuracy at end of each epoch
        model.eval()
        loss_tr, acc_tr = validate_epoch(model, train_subset, collate_batch, subset_size=1000)
        writer.add_scalar("Loss/train", loss_tr, epoch)
        writer.add_scalar("Acc/train", acc_tr, epoch)

        loss_val, acc_val = validate_epoch(model, val_subset, collate_batch, subset_size=1000)
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("Acc/val", acc_val, epoch)
        #  checkpoint
        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = copy.deepcopy(model.state_dict())
        #  early stop check
        if check_early_stop and early_stopper.early_stop(loss_val):
            break
    writer.flush()
    return best_weights


class LSTM2d(nn.Module):
    def __init__(self, embedding_model, vocab, n_classes=139, n_layers=2, hidden_dim=128, lr=0.001, weight_decay=0, p_drop_out=0):

        super().__init__()
        embedding_dim = embedding_model.get_vector(0).shape[0]
        self.embedding = create_embeddings(vocab, embedding_model)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=p_drop_out)

        self.fc = nn.Linear(hidden_dim * 2, n_classes)

        self.lr = lr

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.weight_decay = weight_decay


    def forward(self, text):

        #text = [sent len, batch size]

        embedded = self.embedding(text)

        #embedded = [sent len, batch size, emb dim]

        output, (hidden, cell) = self.lstm(embedded)

        final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        #print(final_hidden_state.shape)

        #final_hidden_state = torch.cat((output[-1], output[-2]))
        #final_hidden_state = torch.cat((hidden[:, -1, :self.hidden_dim], hidden[:, 0, self.hidden_dim:]), dim=1)
        #print(final_hidden_state.shape)
        #assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        res = self.fc(final_hidden_state)
        return res

    def get_optimizer(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
      return optimizer