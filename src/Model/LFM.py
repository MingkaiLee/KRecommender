import pandas as pd
import numpy as np
from typing import Iterable
from torch.optim import SGD
from torch.autograd import Variable
from torch import Tensor
from torch import from_numpy
from torch import matmul
from torch import norm, sum
import logging

__all__ = ['BasicLFM']

class BasicLFM:
    """
    BasicLFM algorithm

    ## Attributes:
        - fit: fit the model with input dataset
        - pred: predict the result of dataset with trained model
    """
    def __init__(self, f: int, lr: float=0.1, ld: float=0.1, *args, **kwargs) -> None:
        """
        Create a basic LFM algorithm model instance.

        ## Parameters:
            - f: int, the num of implicit factors.
            - lr: float, learning rate.
            - ld: float, regularization rate.
        """
        self.user_mat = None
        self.item_mat = None
        self.f = f
        self.lr = lr
        self.ld = ld
    
    def fit(self, train_mat: np.ndarray, scale: tuple, max_iteration: int, loss: str='std'):
        """
        Train the model with train_mat

        ## Parameters:
            - train_mat: np.ndarray, user-item matrix used to train.
            - scale: tuple, (min_score, high_score).
            - max_iteration: int, number of iterations.
            - loss: str, type of loss function, currently support: [std]
        """
        loss_funcs = ['std']
        assert (isinstance(train_mat, np.ndarray)), "Wrong input type, expected ndarray."
        assert (loss in loss_funcs), "Unknown loss function type, please read the function description."
        
        # get valid train data matrix
        mask = np.full(train_mat.shape, fill_value=False, dtype=bool)
        for i in range(train_mat.shape[0]):
            for j in range(train_mat.shape[1]):
                if train_mat[i][j] != 0.0:
                    mask[i][j] = True
        # get score lower bound
        min_score: float = float(scale[0])
        # convert ndarray to Tensor
        mask: Tensor = from_numpy(mask)
        self.user_mat = from_numpy(min_score + np.random.rand(train_mat.shape[0], self.f))
        self.user_mat = Variable(self.user_mat, requires_grad=True)
        self.item_mat = from_numpy(min_score + np.random.rand(self.f, train_mat.shape[1]))
        self.item_mat = Variable(self.item_mat, requires_grad=True)
        train_mat = Variable(from_numpy(train_mat), requires_grad=True)
        # create optimizer
        optimizer = SGD([self.user_mat, self.item_mat], lr=self.lr)
        # train process
        for _ in range(max_iteration):
            # loss function
            loss: Variable = self.__loss_standard(train_mat, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def __loss_standard(self, train_mat, mask) -> Variable:
        # get basic loss
        loss = norm((matmul(self.user_mat, self.item_mat)-train_mat)[mask])**2
        # add regularization item
        loss += self.ld*(sum(norm(self.user_mat, dim=0)**2) + sum(norm(self.item_mat, dim=1)**2))
        return loss

    def pred(self, users: Iterable[int], items: Iterable[int], df: bool=False) -> np.ndarray:
        """
        Predict the user-item matrix with input users list and items list

        ## Paramters:
            - users: Iterable[int], your input user list
            - items: Iterable[int], your input item list
            - df: bool, if True, return a pandas.DataFrame, default False
        """
        assert (self.user_mat is not None), "Model should be trained before used."

        try:
            user_mat = self.user_mat[users,:]
            item_mat = self.item_mat[:,items]
        except:
            logging.error("Your input list may out of range, expected user:[{:d}-{:d}), item:[{:d}-{:d})".format(
                0,
                self.user_mat.shape[0],
                0,
                self.item_mat.shape[1]
            ))
        if df:
            return pd.DataFrame(
                data=matmul(user_mat, item_mat).detach().numpy(),
                index=users,
                columns=items
            )
        return matmul(user_mat, item_mat).detach().numpy()
    
    def reset(self, **kwargs) -> None:
        """
        reset the model parameter
        """
        for key, val in kwargs:
            if hasattr(self, key):
                setattr(self, key, val)
        
        try:
            self.f = int(self.f)
            self.lr = float(self.lr)
            self.ld = float(self.ld)
        except:
            logging.error("Type error, please check your input type.")