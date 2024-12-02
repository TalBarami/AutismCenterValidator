import os
from datetime import datetime
from os import path as osp
import numpy as np
import pandas as pd

from validator.data_handler import DataHandler


class ChildAnnotationData(DataHandler):

    def add(self, idx, ann):
        status, notes, child_ids = ann['status'], ann['notes'], ann['child_ids']
        self.stack.append(idx)
        self.df.loc[idx, ['status', 'notes', 'child_ids', 'timestep']] = [status, notes, str(child_ids), datetime.now()]
        self.save()

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