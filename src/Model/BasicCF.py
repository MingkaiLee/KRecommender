"""
Simple collaborative filtering methods
"""
from collections import defaultdict
import numpy as np

__all__ = ['UserCF', 'ItemCF']

class UserCF:
    """
    Simple User collaborative filtering algorithm

    ## Attributes:
        - fit: fit the model with input dataset
        - pred: predict the result of dataset with trained model 
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Create a UserCF algorithm model instance.
        """
        self.ur_train = None

    def fit(self, ur_train: dict, *args, **kwargs):
        """
        Train the model with train_set
        In this model, it simply prepare the train_set
        """
        self.ur_train = ur_train
    
    def pred(self, ur_test: dict, scale: tuple, k=10, similarity="iif", **kwargs) -> dict:
        """
        Predict outputs of trainset

        ## Parameters:
            - ur_test: dict, the best way to get it is using BasicLoader class
            - scale: tuple with two elements to show the rating scale
            - k: int, the number of most-like users used to recommend items
            - similarity: str, the method of calculating similarity, supports ['iif', 'cos', 'jcd]
        ## Returns：
            - ur_pred: dict, predicted user_matrix
        """
        ur_pred = dict()
        thresh = (scale[0] + scale[1]) / 2
        for u, items in ur_test.items():
            sim_list = list()
            items_set = set([iid for iid, rating in items if rating > thresh])
            for u_o, items_o in self.ur_train.items():
                w = float()
                items_o_set = set([iid for iid, rating in items_o if rating > thresh])
                if similarity == "iif":
                    p = np.log(1+len(items_set.intersection(items_o_set)))
                    c = np.sqrt(len(items_set)*len(items_o_set))
                    w = float(p / c)
                elif similarity == "cos":
                    i = items_set.intersection(items_o_set)
                    c = np.sqrt(len(items_set)*len(items_o_set))
                    w = float(i / c)
                elif similarity == "jcd":
                    i = items_set.intersection(items_o_set)
                    u = items_set.union(items_o_set)
                    w = float(i / u)
                else:
                    raise ValueError("Unexpected similarity mode.")
                sim_list.append((u_o, w))

            sim_list = sorted(sim_list, key=lambda v:v[1], reverse=True)
            # get most-like users' items intersection
            pred_items_set = set()
            for i in range(min(k, len(sim_list))):
                uid = sim_list[i][0]
                pred_items_set = pred_items_set.union(set(
                    [iid for iid, rating in self.ur_train[uid] if rating > thresh]
                ))
            ur_pred[u] = list(pred_items_set)
        
        return ur_pred


class ItemCF:
    """
    Simple Item collaborative filtering algorithm

    ##Attributes:

        - fit: fit the model with input dataset
        - pred: predict the result of dataset with trained model
    """
    def __init__(self) -> None:
        """
        Create a ItemCF algorithm model instance.
        You should not put any args in it
        """
        self.ir_train = None
        pass

    def fit(self, ir_train: dict, *args, **kwargs):
        """
        Train the model with train_set
        In this model, it simply prepare the train_set
        """
        self.ir_train = ir_train
    
    def pred(self, ir_test: dict, scale: tuple, k=10, similarity="", *args, **kargs) -> dict:
        """
        Predict outputs of trainset

        ## Parameters:
            - ir_test: dict, the best way to get it is using BasicLoader class
            - scale: tuple with two elements to show the rating scale
            - k: int, the number of most-like users used to recommend items
            - similarity: str, the method of calculating similarity, supports ['iif', 'cos', 'jcd]
        ## Returns：
            - ir_pred: dict, predicted user_matrix
        """
        # TODO: Realize this simple algorithm
        pass
    

