import os
from datetime import datetime
from os import path as osp

import numpy as np
import pandas as pd
from skeleton_tools.utils.tools import write_json


class DataHandler:
    def __init__(self, annotations_file, annotations_dir, data_dir):
        self.annotations_file = annotations_file
        self.annotations_dir = annotations_dir
        self.data_dir = data_dir
        self.df = self.collect_annotations()
        self.save()
        self.stack = []

    def collect_annotations(self):
        files = [osp.join(self.annotations_dir, f) for f in os.listdir(self.annotations_dir)]
        df = pd.concat([pd.read_csv(f) for f in files])
        df = df[df['tracked'] == 1].reset_index(drop=True)
        df = df.drop(columns=['tracked'])
        df['status'] = np.nan
        df['notes'] = np.nan
        df['child_ids'] = np.nan
        df['timestep'] = np.nan
        df['segment_name'] = df.apply(lambda row: f"{row['basename']}_{row['start_frame']}_{row['end_frame']}", axis=1)
        df['video_path'] = df['segment_name'].apply(lambda s: osp.join(self.data_dir, f'{s}.mp4'))
        df['data_path'] = df['segment_name'].apply(lambda s: osp.join(self.data_dir, f'{s}.pkl'))
        if osp.exists(self.annotations_file):
            _df = pd.read_csv(self.annotations_file)
            merged = df.merge(_df, on=['basename', 'start_frame', 'end_frame'], how='inner', suffixes=('', '_duplicate'))
            common_indices = merged.index
            df = df.drop(common_indices)
            df = pd.concat([_df, df])
        return df.sort_values(by=['start_frame', 'basename']).reset_index(drop=True)

    def save(self):
        self.df = self.df.sort_values(by=['start_frame', 'basename'])
        self.df.to_csv(self.annotations_file, index=False)

    def revert(self):
        if len(self.stack) == 0:
            return
        self.df.loc[self.stack.pop(), ['status']] = np.nan
        self.save()

    def get_rows(self):
        return self.df[self.df['status'].isna()]

    def add(self, idx, status, notes, child_ids):
        self.stack.append(idx)
        self.df.loc[idx, ['status', 'notes', 'child_ids', 'timestep']] = [status, notes, str(child_ids), datetime.now()]
        self.save()
