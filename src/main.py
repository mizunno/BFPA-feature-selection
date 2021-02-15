from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src.BFPA import BinaryFlowerPollinationAlgorithm
from src.datasets_dispatcher import DatasetDispatcher
import pandas as pd
import numpy as np
import random

random_seed = 1993


def hyperparameters_search(x_train, y_train, x_val, y_val, selected_features=None):
    hyperparameters_dict = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None] + list(range(2, 10, 2)),
        "min_samples_split": list(range(2, 10, 1)),
        "min_samples_leaf": list(range(1, 10, 1)),
        "max_features": [None, "auto", "log2"]
    }

    estimator = DecisionTreeClassifier()
    grid = GridSearchCV(estimator, hyperparameters_dict, n_jobs=-1)

    if selected_features is not None:
        x_train = x_train[:, selected_features]
        x_val = x_val[:, selected_features]

    grid.fit(x_train, y_train)
    score = grid.score(x_val, y_val)

    return score


if __name__ == "__main__":
    np.random.seed(random_seed)
    random.seed(random_seed)

    dataset_dispatcher = DatasetDispatcher("../datasets/")
    experiments_results = pd.DataFrame(columns=["iteration",
                                                "dataset",
                                                "num_features",
                                                "num_selected_features",
                                                "base_score",
                                                "selected_features_score",
                                                "num_flowers"])

    datasets = ["BASEHOCK", "GLI_85", "PCMAC", "RELATHE", "SMK_CAN_187", "TOX_171", "AR10P"]

    for i in range(20):
        print("######################################################################################")
        print("############################# Iteration " + str(i) + "################################")
        print("######################################################################################")
        for dataset in datasets:
            print("Dataset: " + dataset)

            x_train, x_val, y_train, y_val = dataset_dispatcher.get_dataset(name=dataset,
                                                                            test_size=0.33,
                                                                            make_split=True,
                                                                            random_seed=random_seed)

            print("Num features: ", str(x_train.shape[1]))

            base_model = DecisionTreeClassifier()
            base_model.fit(x_train, y_train)
            base_score = base_model.score(x_val, y_val)
            print("Base score:" + str(base_score))

            num_flowers = 20
            optimizer = BinaryFlowerPollinationAlgorithm(num_flowers, x_train.shape[1], DecisionTreeClassifier())
            selected_features_score, selected_features = optimizer.fit(x_train=x_train,
                                                                       y_train=y_train,
                                                                       x_val=x_val,
                                                                       y_val=y_val)

            print("Selected features score " + str(selected_features_score))
            print("Num selected features " + str(sum(selected_features)))

            row = {
                "iteration": str(i),
                "dataset": dataset,
                "num_features": x_train.shape[1],
                "num_selected_features": sum(selected_features),
                "base_score": base_score,
                "selected_features_score": selected_features_score,
                "num_flowers": num_flowers
            }

            experiments_results = experiments_results.append(row, ignore_index=True)

    experiments_results.to_csv("experiments_results_3.csv", index=False)