import os
import pandas as pd
import scipy.sparse as sparse
import numpy as np


class DataPreparation(object):
    def __init__(self):
        self.df = None
        self.m = None

    def read_items_file(self, file_path):
        member_items_df = pd.read_csv(os.path.join(file_path, 'items'))
        member_items_df = member_items_df[['member_id', 'item_code', 'quantity']]

        self.df = member_items_df

    def pre_processing(self):
        self.df = self.df.loc[self.df.member_id.nonull() & self.df.item_code.nonull()]

    def aggregate_items(self):
        self.df = self.df.groupby(['member_id', 'item_code'])\
            .sum()\
            .reset_index()

        self.df.quantity.loc[self.df.quantity == 0] = 1
        self.df = self.df[self.df.quantity > 0]

    def df_to_matrix(self):
        members = list(np.sort(self.df.member_id.unique()))  # Get our unique customers
        items = list(self.df.item_code.unique())  # Get our unique products that were purchased
        quantity = list(self.df.quantity)  # All of our purchases

        rows = self.df.member_id.astype('category', categories=members).cat.codes
        cols = self.df.StockCode.astype('category', categories=items).cat.codes
        self.m = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(members), len(items)))
        print(self.m.shape)

    def calc_sparsity(self):
        matrix_size = self.m.shape[0] * self.m.shape[1]
        num_purchases = len(self.m.nonzero()[0])
        sparsity = 100*(1 - (num_purchases/matrix_size))

        return sparsity
