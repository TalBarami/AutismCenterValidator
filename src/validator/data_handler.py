import os
from abc import abstractmethod, ABC
from datetime import datetime
from os import path as osp

import numpy as np
import pandas as pd


class DataHandler(ABC):
    def __init__(self, annotator_id):
        self.annotator_id = annotator_id
        self.df = self.collect_annotations()
        # vs = self.df['start_frame'].unique()
        # self.group = vs[self.annotator_id::2]
        # self.idx = self.df[self.df['start_frame'].isin(self.group)].index
        self.to_annotate = self.df[self.df['status'].isna()].index
        self.idx = self.to_annotate[:len(self.to_annotate)//2] if self.annotator_id == 0 else self.to_annotate[len(self.to_annotate)//2:]
        self.save()
        self.stack = []

    @abstractmethod
    def collect_annotations(self):
        pass

    def save(self):
        self.df = self.df.sort_values(by=['start_frame', 'basename'])
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
