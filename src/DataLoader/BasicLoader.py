# Creatd by limingkai on 21.11.06
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from scipy import sparse

from .DataSet import *

__all__ = ['BasicLoader']

class BasicLoader(object):
    """Basic data manager,
    convert and split the input data to the type which could be trained and validated.

    ## attributes:
        - load_df(): load data from pandas DataFrame.
        - raw_date: the raw data, list[tuple].
        - rating_scale: legal rating scale.
        - train_test_split(): get the train and test user/item matrixes.
        - train_mat(): get the user-item matrix. 
    """
    def __init__(self, data=None) -> None:
        """Create a basic data loader to help model train and evaluating

        ## Parameters:
            - data: list of tuples with four elements, (uid, iid, rating, timestamp)
        """
        super().__init__()
        self.__raw_data = data
        try:
            self.__rating_scale = self.__get_rating_scale()
        except TypeError:
            self.__rating_scale = ()
    
    def __get_rating_scale(self) -> tuple:
        min_bound = self.__raw_data[0][2]
        max_bound = self.__raw_data[0][2]
        for _, _, r, _ in self.__raw_data:
            if r < min_bound:
                min_bound = r
            if r > max_bound:
                max_bound = r
        return (min_bound, max_bound)
    
    @property
    def raw_data(self):
        return self.__raw_data
    
    @property
    def rating_scale(self):
        return self.__rating_scale
    
    def load_df(self, df: pd.DataFrame) -> None:
        """load data from pandas DataFrame
        
        prepare your input DataFrame with four colums:
        (user_id, item_id, rating, timestamp)
        """
        self.__raw_data = [(uid, iid, float(r), None)
                            for (uid, iid, r, _) in
                            df.itertuples(index=False)]
        self.__rating_scale = self.__get_rating_scale()
    
    def __construct_data_set(self) -> DataSet:
        """Transform list raw train_set to TrainSet instance

        most of codes here are from `<scikit-surprise>` , it's github repository URL:

        https://github.com/NicolasHug/Surprise
        """
        raw2inner_id_users: dict = dict()
        raw2inner_id_items: dict = dict()

        current_u_index = 0
        current_i_index = 0

        ur: defaultdict = defaultdict(list)
        ir: defaultdict = defaultdict(list)

        for raw_uid, raw_iid, r, _ in self.__raw_data:
            try:
                uid = raw2inner_id_users[raw_uid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[raw_uid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[raw_iid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[raw_iid] = current_i_index
                current_i_index += 1
            
            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        return DataSet(
            ur=ur,
            ir=ir,
            n_users=len(ur),
            n_items=len(ir),
            n_ratings=self.__rating_scale,
            rating_scale=len(self.__raw_data),
            raw2inner_id_users=raw2inner_id_users,
            raw2inner_id_items=raw2inner_id_items
        )
    
    def train_mat(self) -> np.ndarray:
        """
        Convert raw data into matrix which used by LFM algorithms

        TODO(mingkai): return sparse matrix to save memory.
        """
        assert (self.__raw_data != None), 'No data loaded before.'

        self.__data_set = self.__construct_data_set()
        mat = np.full(
            shape=(self.__data_set.n_users, self.__data_set.n_items),
            fill_value=0.0,
            dtype=float
        )
        for uid, item_list in self.__data_set.ur.items():
            for iid, rating in item_list:
                mat[uid, iid] = rating
        
        return mat

    def train_trans_mat(self) -> tuple:
        """
        Convert raw data into sparse matrix used by Graph-based algorithms

        TODO(mingkai): return sparse matrix to save memory.

        ## Returns:
            - trans_mat: np.ndarray, transition matrix which could be used by PersonalRank algorithm.
            - user_num: int
            - item_num: int
        """
        assert (self.__raw_data != None), 'No data loaded before.'

        def calcu_out(user_vec: np.ndarray):
            liked = 0
            res = np.full(shape=user_vec.shape, fill_value=0.0, dtype=float)
            for v in user_vec:
                if v > np.average(self.__rating_scale):
                    liked += 1
            for i, v in enumerate(user_vec):
                if v > np.average(self.__rating_scale):
                    res[i] = 1 / liked
            return res


        # get raw rating train mat
        raw_mat = self.train_mat()
        # change raw rating mat to probability matrix
        raw_u_mat = np.apply_along_axis(calcu_out, 1, raw_mat)
        raw_i_mat = np.apply_along_axis(calcu_out, 1, np.transpose(raw_mat, axes=(1, 0)))
        raw_u_mat = np.hstack((np.full((raw_u_mat.shape[0], raw_u_mat.shape[0]), 0, dtype=float), raw_u_mat))
        raw_i_mat = np.hstack((raw_i_mat, np.full((raw_i_mat.shape[0], raw_i_mat.shape[0]), 0, dtype=float)))
        raw_mat = np.vstack((raw_u_mat, raw_i_mat))

        # return sparse.dia_matrix(raw_mat)
        return (raw_mat, raw_u_mat.shape[0], raw_i_mat.shape[0])
    
    def train_test_split(self, rating=0.8, shuffle=True):
        """split data into train_set and test_set

        ## Parameters
            - rating: ratio of the number of training sets to the number of test sets
            - shuffule: whether or not to shuffle the data before splitting
        ## Returns
            - ur_train: user matrix used for trainning
            - ur_test: user matrix used for testing
            - ir_train: item matrix used for trainning
            - ir_test: item matrix used for testing
        """
        assert (self.__raw_data != None), 'No data loaded before.'

        self.__data_set = self.__construct_data_set()

        ur_train = dict()
        ur_test = dict()
        ir_train = dict()
        ir_test = dict()
        
        if shuffle:
            # split train and test ur
            ur_keys = set(self.__data_set.ur)
            ur_train_keys = set(random.sample(ur_keys, k=int(len(ur_keys)*rating)))
            ur_test_keys = ur_keys - ur_train_keys

            for key in ur_train_keys:
                ur_train[key] = self.__data_set.ur[key]
            for key in ur_test_keys:
                ur_test[key] = self.__data_set.ur[key]
            
            # split train and test ir
            ir_keys = set(self.__data_set.ir)
            ir_train_keys = set(random.sample(ir_keys, k=int(len(ir_keys)*rating)))
            ir_test_keys = ir_keys - ir_train_keys

            for key in ir_train_keys:
                ir_train[key] = self.__data_set.ir[key]
            for key in ir_test_keys:
                ir_test[key] = self.__data_set.ir[key]
        else:
            # split train and test ur
            train_size = int(len(self.__data_set.ur)*rating)
            for i, key in enumerate(self.__data_set.ur.keys()):
                if i < train_size:
                    ur_train[key] = self.__data_set.ur[key]
                else:
                    ur_test[key] = self.__data_set.ur[key]
            
            # split train and test ir
            train_size = int(len(self.__data_set.ir)*rating)
            for i, key in enumerate(self.__data_set.ir.keys()):
                if i < train_size:
                    ir_train[key] = self.__data_set.ir[key]
                else:
                    ir_test[key] = self.__data_set.ir[key]

        return ur_train, ur_test, ir_train, ir_test
    
    def k_fold(self) -> object:
        assert (self.__raw_data != None), 'No data loaded before.'
        """
        TODO(MingkaiLee): K-fold validation dataset partition.
        """
        pass