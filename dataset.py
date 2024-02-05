import pandas as pd
import numpy as np

from knn.custom_errors import FeatureLabelSizeMissmatch, LabelNotFound, InsufficientData
from typing import Optional, Tuple
from pathlib import Path


class Dataset:
    """
    General dataset class to hold features and labels. It works as validator and data holder.
    """
    def __init__(self, 
                 X: pd.DataFrame or pd.Series, 
                 y: pd.Series):
         """
         Parameters:
         -----------
         X: (pd.DataFrame or pd.Series) Feature set, with columns as the name of the feature
         y: (pd.Series) Label set
         """
         Dataset.__dataset_validation(X, y)
         self.__X = X
         self.__y = y
    
    def __str__(self):
        out = f"""
            FEATURES: {self.feature_names}
            DATASET_CARDINALITY: {self.cardinality}
        """
        return out

    def __repr__(self):
        out = f"FEATURES: {self.feature_names}\n"
        f"LABELS: {self.label_names}\nDATASET_CARDINALITY: {self.cardinality}"
        return out
        
    @staticmethod
    def shuffle(X, y, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Shuffles the features and labels randomly
        
        Parameters:
        -----------
        random_state: (int) Reproduces the same shuffle for same integer. By default, None
        X: (pd.DataFrame or pd.Series) Feature set, with columns as the name of the feature
        y: (pd.Series) Label set

        Returns:
        --------
        (tuple) Shuffled X and y
        """
        label_nm = y.name
        X[label_nm] = y
        if random_state:
            np.random.seed(random_state)
            X = X.sample(n=len(X), random_state=random_state).reset_index(drop=True)
        else:
            X = X.sample(n=len(X), random_state=np.random.randint(0, 10e5, 1)).reset_index(drop=True)

        y = X.pop(label_nm)
        y.name = label_nm

        return X, y

    @property
    def X(self):
        return self.__X

    @property
    def feature_names(self):
        return list(self.__X.columns)

    @property
    def y(self):
        return self.__y

    @property
    def cardinality(self):
        return self.__X.shape[0]
    
    @property
    def label_names(self):
        return self.__y.unique().tolist()

    def __dataset_validation(features, labels):
        Dataset.__features_validation(features)
        Dataset.__labels_validation(labels)
        
        if features.shape[0] != labels.shape[0]:
            raise FeatureLabelSizeMissmatch("Cardinality of features and labels are not same!")

    def __features_validation(features):
        if isinstance(features, (pd.DataFrame, pd.Series)):
            if sum(features.isnull().any()):
                raise ValueError("Features contain null values")
            elif features.apply(pd.api.types.is_numeric_dtype, axis=0).sum() != features.shape[1]:
                raise ValueError("Features contains non numeric values")
        else:
            raise TypeError("Only pd.DataFrame or pd.Series object are allowed for features!")
    
    def __labels_validation(labels):
        if isinstance(labels, pd.Series):
            if labels.isnull().any():
                raise ValueError("Labels contain null values")
        else:
            raise TypeError("Only pd.Series object are allowed for Labels!")

    def split(
            self, 
            shuffle: bool = False, 
            random_state: Optional[int] = None,
            fraction: float = 0.8
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the given dataset and returns X_train, X_test, y_train, y_test
        Parameters:
        -----------
        shuffle: (bool) Whether to shuffle the given features and labels, by default, False
        random_state: (int) Reproduces the same shuffle for same integer. By default, None
        fraction: (float) how much to split between train and test, default is 0.8
        """
        if shuffle:
            X, y = Dataset.shuffle(self.__X.copy(), self.__y.copy(), random_state)
        else:
            X, y = self.__X.copy(), self.__y.copy()
        
        dataset = Dataset.combine_dataset(X, y)
        
        if dataset.shape[0] < 10:
            raise InsufficientData("Not enough data to split, minimum cardinality is 10")
        
        if not (0 <= fraction <= 1):
            raise ValueError("fraction value should be between 0 and 1")

        train_size = int(dataset.shape[0] * fraction)
        X_train = dataset.loc[:train_size, :]
        X_test = dataset.loc[train_size:, :]
        
        y_train = X_train.pop(y.name)
        y_test = X_test.pop(y.name)

        return X_train, X_test, y_train, y_test
        
        
    @staticmethod
    def combine_dataset(X, y) -> pd.DataFrame:
        """
        Combines the features and labels in a single dataframe
        Parameters:
        -----------
        X: (pd.DataFrame or pd.Series) Feature set, with columns as the name of the feature
        y: (pd.Series) Label set
        Returns:
        --------
        X and y Combined dataframe
        """
        label_nm = y.name
        X[label_nm] = y

        return X.copy()

def csv_dataset(path: str or Path, 
                label_column: Optional[str] = None) -> Dataset:
    """
    Load dataset from the csv file.
    Parameters:
    -----------
    path: (str or pathlib.Path) csv file path
    label_column: (str) Column name to consider as labels, by default it is None, ie, it will consider the last column as label.

    Returns:
    --------
    Dataset object
    """
    features = pd.read_csv(path)
    if label_column:
        if label_column in features.columns:
            labels = features.pop(label_column)
        else:
            raise LabelNotFound()
    else:
        labels = features.pop(features.columns[-1])
    
    return Dataset(features, labels)
