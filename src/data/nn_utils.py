import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
import random
from torch.utils.data import Sampler
from torchtext.data.utils import get_tokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = get_tokenizer('basic_english')


class MoviePLotsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.labels[idx], self.texts[idx]


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def get_vocab(train_datapipe, tokenizer, min_freq=5):
  vocab = build_vocab_from_iterator(yield_tokens(train_datapipe, tokenizer),
                                    specials=['<UNK>', '<PAD>'],
                                    min_freq=min_freq,
                                    special_first=True)
  vocab.set_default_index(vocab['<UNK>'])
  return vocab
  



class BatchSamplerSimilarLength(Sampler):
  def __init__(self, dataset, batch_size, indices=None, shuffle=True):
    self.batch_size = batch_size
    self.shuffle = shuffle
    # get the indices and length
    self.indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(dataset)]
    # if indices are passed, then use only the ones passed (for ddp)
    if indices is not None:
       self.indices = torch.tensor(self.indices)[indices].tolist()

  def __iter__(self):
    if self.shuffle:
       random.shuffle(self.indices)

    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(self.indices), self.batch_size * 100):
      pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
    self.pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    batches = [self.pooled_indices[i:i + self.batch_size] for i in
               range(0, len(self.pooled_indices), self.batch_size)]

    if self.shuffle:
        random.shuffle(batches)
    for batch in batches:
        yield batch

  def __len__(self):
    return len(self.pooled_indices) // self.batch_size


def create_embeddings(vocab, embedding_model):
    embedding_size = embedding_model.vectors.shape[1]
    vocab_size = len(vocab)
    weights_matrix = np.zeros((vocab_size, embedding_size))
    for i, word in enumerate(vocab.get_itos()):
        try:
            # if word exists in embedding matrix then use it
            weights_matrix[i] = embedding_model.get_vector(word)
        except KeyError:
            # if word does not exist in embedding matrix use vector sampled from normal distribution
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_size,))
    # init <unk> and <pad> with zeros
    weights_matrix[0, :] = np.zeros(shape=(embedding_size,))
    weights_matrix[1, :] = np.zeros(shape=(embedding_size,))
    return nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix).type(torch.float32)).to(device)


class MovieHFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, label2id=None):
        self.encodings = encodings
        self.labels = [int(label2id[label]) for label in labels]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    

class MovieHFDatasetMLL(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, label2id=None):
        self.encodings = encodings
        #self.labels = [[int(label2id[label]) for label in targets_of_instance] for targets_of_instance in labels]
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_whole_target(data):
    dataloader = dataloader(data, batch_size=100)
    target = []
    for _, y in dataloader:
        target.append(y.cpu().detach().numpy())
    return np.array(target)