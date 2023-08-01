from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import optuna
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
import os
load_dotenv()
SEED=int(os.getenv('SEED'))


def tune_hyperparams_optuna(objective, n_trials, metric: str):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)
    print(f'Best {metric} score: {study.best_value}')
    print(f'Best Params: {study.best_params}')
    return study


def tune_hyperparams_rand(clf, params_dict, X_train, y_train, n_trials=10):
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=params_dict,
                                       scoring='f1_weighted',
                                       cv=3,
                                       n_iter=n_trials,
                                       random_state=SEED,
                                       verbose=2)
    random_search.fit(X_train, y_train)
    # Print the best hyperparameters and the corresponding score
    print('Best hyperparameters:', random_search.best_params_)
    print('Best f1 score:', random_search.best_score_)
    return random_search.best_estimator_

