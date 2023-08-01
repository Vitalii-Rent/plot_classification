import string
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
import os
load_dotenv()
SEED=os.getenv('SEED')


def split_transform_data(X, y, vectorizer):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, shuffle=True, random_state=SEED, stratify=y)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test











  
