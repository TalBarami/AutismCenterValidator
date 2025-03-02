import os
from abc import abstractmethod, ABC
from datetime import datetime
from os import path as osp

import numpy as np
import pandas as pd


class DataHandler(ABC):
    def __init__(self, annotator_id, annotations_file):
        self.annotator_id = annotator_id
        self.annotations_file = annotations_file
        self.df = self.collect_annotations()
        # subset = pd.read_csv(osp.join(self.annotations_dir, '..', 'sample12.csv'))['basename'].unique()
        # self.to_annotate = self.df[self.df['status'].isna() & self.df['basename'].isin(subset)].index
        # self.idx = self.to_annotate[:len(self.to_annotate)//2] if self.annotator_id == 0 else self.to_annotate[len(self.to_annotate)//2:]

        self.idx = self.get_indices()
        self.save()
        self.stack = []

    def get_indices(self):
        return self.df[self.df['status'].isna()].index

    @abstractmethod
    def collect_annotations(self):
        pass

    def save(self):
        # self.df = self.df.sort_values(by=['start', 'assessment'])
        self.df.to_csv(self.annotations_file, index=False)

    def revert(self):
        if len(self.stack) == 0:
            return
        self.df.loc[self.stack.pop(), ['status']] = np.nan
        self.save()

    def get_rows(self):
        df = self.df.loc[self.idx]
        return df[df['status'].isna()]

    @abstractmethod
    def add(self, idx, row_info):
        pass
