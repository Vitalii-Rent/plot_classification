from torch.utils.data import DataLoader
import sys
sys.path.append('src')
from data.nn_utils import BatchSamplerSimilarLength
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from transformers import EvalPrediction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, data, collate_batch):
    dataloader = DataLoader(data,
                            batch_sampler=BatchSamplerSimilarLength(
                                dataset=data,
                                batch_size=100),
                            collate_fn=collate_batch)
    n_batches = total_acc = 0
    total_correct = 0
    total_instances = 0
    # total = 0
    # it = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = torch.argmax(model(X), dim=1)
            # print(pred)
            # print(y)
            correct_predictions = sum(pred == y).item()
            total_correct += correct_predictions
            total_instances += len(y)
            # acc = accuracy(pred, y).item()
            # total += acc
            # it += 1
    mean_acc = round(total_correct / total_instances, 3)
    print(f'Accuracy score = {mean_acc}')
    # print(f'Accuracy score = {acc/it}  {total_correct/total_instances}')
    return mean_acc


def get_np_targets(model, data, collate_batch):
    all_preds = []
    all_true = []
    dataloader = DataLoader(data,
                            batch_sampler=BatchSamplerSimilarLength(
                                dataset=data,
                                batch_size=100),
                            collate_fn=collate_batch)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = torch.argmax(model(X), dim=1).detach().cpu().numpy()
            all_preds.append(pred)
            all_true.append(y.detach().cpu().numpy())
    return np.concatenate([arr for arr in all_preds]), np.concatenate([arr for arr in all_true])


def compute_metrics_mlc(p: EvalPrediction):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')

    return {"accuracy": accuracy,  "f1": f1}



    
    
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'weighted')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


def transform_predictions(predictions, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    return y_pred