import os
from datetime import datetime
from os import path as osp

import numpy as np
import pandas as pd
from skeleton_tools.utils.tools import write_json


class DataHandler:
    def __init__(self, annotations_file, annotations_dir, data_dir, annotator_id):
        self.n_annotators = 2
        self.annotations_file = annotations_file
        self.annotations_dir = annotations_dir
        self.data_dir = data_dir
        self.annotator_id = annotator_id
        self.df = self.collect_annotations()
        # vs = self.df['start_frame'].unique()
        # self.group = vs[self.annotator_id::2]
        # self.idx = self.df[self.df['start_frame'].isin(self.group)].index
        self.to_annotate = self.df[self.df['status'].isna()].index
        self.idx = self.to_annotate[:len(self.to_annotate)//2] if self.annotator_id == 0 else self.to_annotate[len(self.to_annotate)//2:]
        self.save()
        self.stack = []

    def collect_annotations(self):
        if osp.exists(self.annotations_file):
            _df = pd.read_csv(self.annotations_file)
        else:
            _df = pd.DataFrame(columns=['basename', 'start_frame', 'end_frame', 'fps', 'frame_count', 'status', 'notes', 'child_ids', 'timestep', 'segment_name', 'video_path', 'data_path'])
        ann = _df['basename'].unique()
        files = [osp.join(self.annotations_dir, f) for f in os.listdir(self.annotations_dir) if '_'.join(f.split('_')[:6]) not in ann]
        _app = pd.concat([pd.read_csv(f) for f in files])
        _app = _app[_app['tracked'] == 1].reset_index(drop=True)
        _app = _app.drop(columns=['tracked'])
        _app['status'] = np.nan
        _app['notes'] = np.nan
        _app['child_ids'] = np.nan
        _app['timestep'] = np.nan
        _app['segment_name'] = _app.apply(lambda row: f"{row['basename']}_{row['start_frame']}_{row['end_frame']}", axis=1)
        # _app['video_path'] = _app['segment_name'].apply(lambda s: osp.join(self.data_dir, f'{s}.mp4'))
        # _app['data_path'] = _app['segment_name'].apply(lambda s: osp.join(self.data_dir, f'{s}.pkl'))
        merged = _app.merge(_df, on=['basename', 'start_frame', 'end_frame'], how='inner', suffixes=('', '_duplicate'))
        common_indices = merged.index
        df = _app.drop(common_indices)
        df = pd.concat([_df, df])
        df = df.sort_values(by=['start_frame', 'basename']).reset_index(drop=True)
        df['video_path'] = df['segment_name'].apply(lambda s: osp.join(self.data_dir, f'{s}.mp4'))
        df['data_path'] = df['segment_name'].apply(lambda s: osp.join(self.data_dir, f'{s}.pkl'))
        return df


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

    def add(self, idx, status, notes, child_ids):
        self.stack.append(idx)
        self.df.loc[idx, ['status', 'notes', 'child_ids', 'timestep']] = [status, notes, str(child_ids), datetime.now()]
        self.save()
