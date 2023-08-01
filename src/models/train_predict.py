import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score, hamming_loss, f1_score


def count_score_labels(y_pred, y_true, labels, all_y_true=None):
    if all_y_true is None:
        counts = np.sum(y_true == 1, axis=0)
    else:
        counts = np.sum(all_y_true == 1, axis=0)
    f1_scores = f1_score(y_pred, y_true, average=None)
    label_scores = pd.DataFrame(np.array([f1_scores, counts]).T, columns=['f1_score', 'n_instances'], index=labels)
    return label_scores




def evaluate_ml(true, pred, data_type=None):
    print(f"model's {data_type} score = {accuracy_score(true, pred)}")
    print(f"model's jaccard {data_type} score = {jaccard_score(true, pred, average='samples')}")
    print(f"model's hamming {data_type} loss = {hamming_loss(true, pred)}")


def train_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    print(f"model's train acc score = {model.score(X_train, y_train)}")
    print(f"model's test acc score = {model.score(X_test, y_test)}")
    print(f"model's train f1 score = {f1_score(y_train, model.predict(X_train), average='weighted')}")
    print(f"model's test f1 score = {f1_score(y_test, model.predict(X_test), average='weighted')}")
    return model

def one_match(y_pred, y_true):
    matches = [(true_labels.any() and (true_labels.astype(int) & pred_labels.astype(int)).any()) for true_labels, pred_labels in zip(y_true, y_pred)]
    return sum(matches) / len(matches)


def train_evaluate_mll(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions_tr = model.predict(X_train)
    predictions_test = model.predict(X_test)
    print(f"model's train score = {model.score(X_train, y_train)}")
    print(f"model's test score = {model.score(X_test, y_test)}")
    print(f"model's jaccard train score = {jaccard_score(y_train, predictions_tr, average='samples')}")
    print(f"model's hamming train loss = {hamming_loss(y_train, predictions_tr)}")
    print(f"model's jaccard test score = {jaccard_score(y_test, predictions_test, average='samples')}")
    print(f"model's hamming test loss = {hamming_loss(y_test, predictions_test)}")
    print(f"model's one match test score = {one_match(y_test, predictions_test)}")
    return model