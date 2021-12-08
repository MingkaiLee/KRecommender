from typing import Iterable
import torch
import numpy as np
import numpy.linalg as linalg
import logging
from scipy import sparse

__all__ = ['PersonalRank']

class PersonalRank:
    """
    PersonalRank algorithm

    ## Attributes:
        - fit: fit the model with input dataset
        - pred: predict the result of dataset with trained model
    """

    def __init__(self, alpha, *args, **kwargs) -> None:
        """
        Create a basic PersonalRank algorithm model instance

        ## Attributes:
            - alpha: float, the probability to continue walking, hyper-parameter
        """
        self.alpha = alpha
        self.train_mat = None
        self.user_num = 0
        self.item_num = 0

    def fit(self, trans_mat: np.ndarray, user_num: int, item_num: int, *args, **kwargs):
        """
        Trian the model with trans_mat

        For I used the no-iteration approach, it simply prepare the train_mat
        """
        self.train_mat = trans_mat
        self.user_num = user_num
        self.item_num = item_num
    
    def pred(self, users: Iterable[int]=[], items: Iterable[int]=[], *args, **kwargs):
        """
        Predict the user-item matrix with input users list

        ## Parameters:
            - users: Iterable[int], your input user list
            - items: Iterable[int], your input item list
        """
        assert (self.train_mat is not None), "Model should be trained before used."

        # create r mat
        try:
            user_mat = np.full((self.train_mat.shape[0], len(users)), fill_value=0, dtype=float)
            for i, v in enumerate(users):
                user_mat[i, v] = 1
            item_mat = np.full((self.train_mat.shape[0], len(items)), fill_value=0, dtype=float)
            for i, v in enumerate(items):
                item_mat[i, v] = 1
        except:
            logging.error("Your input list may out of range, expected user:[{:d}-{:d}), item:[{:d}-{:d})".format(
                0,
                self.user_num,
                0,
                self.item_num
            ))
        # construct linear equations
        A = np.eye(self.train_mat.shape[0]) - self.alpha * np.transpose(self.train_mat, (1, 0))
        # solve user
        b = (1 - self.alpha) * user_mat
        r_u = linalg.solve(A, b)
        # solve item
        b = (1 - self.alpha) * item_mat
        r_i = linalg.solve(A, b)

        return r_u, r_i