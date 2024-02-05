import pandas as pd
import numpy as np

from typing import Literal, List
from numbers import Real


class KNNClassifier:
    """
    Specialized KNN classifier designed to increase the speed of the exicution by using multiprocessing and GPU.
    """
    def __init__(self,
                 k: int = 5,
                 p: int = 2,
                 weights: Literal['uniform', 'weighted'] = 'uniform',
                 size: int = 3,
                 buckets: int = 2) -> None:
        self.__k = k
        self.__p = p
        self.__weights = weights
        self.__size = size
        self.__bucket_value_checker(buckets)
        self.__dataset = {}

    def __bucket_value_checker(self, buckets: int):
        if not isinstance(buckets, int):
            raise TypeError("buckets needs to be int")
        if buckets > 5:
            raise ValueError("buckets cannot be greater than 5")
        self.__buckets = buckets

    @staticmethod
    def __valid_task(task):
        if isinstance(task, str):
            task = task.lower()
            if task not in ['classification', 'regression']:
                raise ValueError("Wrong task added!")
            else:
                return task
        else:
            raise TypeError("Task is of wrong type!")
    
    @property
    def nearest_neighbors(self):
        return self.__k
    
    @property
    def buckets(self):
        return self.__buckets
    
    @property
    def weights(self):
        return self.__weights

    @property
    def size(self):
        return self.__size
    
    def __dataset_splitter(self, features, labels):
        features['labels'] = labels
        each_size = int(features.shape[0] / self.__buckets)
        i_i = 0
        i_f = each_size
        for i in range(self.buckets):
            self.__dataset[i] = features.loc[i_i: i_f, :]
            i_i, i_f = int(i_f), int(i_f * 2)

    
    def train(self, features: pd.DataFrame or pd.Series, labels: pd.Series) -> None:
        self.__features = features
        self.__labels = labels
        self.__dataset_splitter(features, labels)
        print("Training Done!")

    def distance_matrics(self, p1: int, X: pd.DataFrame, p: int):
        X.loc[:, 'summation'] = np.sum(((X.iloc[:, :-1] - p1) ** p) ** (1/p), axis=1)
        return X
    
    def __inference(self, query_point, X, p):
        X = self.distance_matrics(query_point.copy(), X.copy(), p)
        X = X.sort_values(by='summation')[:self.__k]
        
        return X
    
    def inference(self, query_point: List[Real]) -> dict:
        """
        Runs the inference on the given model
        """
        o = []
        for i in self.__dataset:
            o.append(self.__inference(query_point, self.__dataset[i], self.__p))
        
        final = pd.concat([*o]).sort_values(by='summation')

        return {"prediction": final.iloc[0, -2], "distance": final.iloc[0, -1]}

