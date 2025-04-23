import os
from datetime import datetime
from os import path as osp
import numpy as np
import pandas as pd

from validator.data_handler import DataHandler




class SkeletonAnnotationData(DataHandler):
    def __init__(self, annotator_id, test_dir):
        self.test_dir = test_dir
        self.videos_dir = osp.join(self.test_dir, 'videos')
        super().__init__(annotator_id, osp.join(test_dir, 'annotations.csv'))

    def add(self, idx, ann):
        status, notes = ann['status'], ann['notes']
        self.stack.append(idx)
        self.df.loc[idx, ['status', 'notes', 'timestamp']] = [status, notes, datetime.now()]
        self.save()

    def collect_annotations(self):
        if osp.exists(self.annotations_file):
            _df = pd.read_csv(self.annotations_file)
        else:
            _df = pd.DataFrame(columns=['basename', 'start_frame', 'end_frame', 'fps', 'frame_count', 'status', 'notes', 'timestamp', 'segment_name', 'video_path'])
        ann = _df['segment_name'].unique()
        files = [osp.splitext(f)[0] for f in os.listdir(self.videos_dir) if osp.splitext(f)[0] not in ann]
        _app = pd.DataFrame(columns=_df.columns)
        _app['segment_name'] = files
        _app['basename'] = _app['segment_name'].apply(lambda f: '_'.join(f.split('_')[:6]))
        _app['start_frame'] = _app['segment_name'].apply(lambda f: f.split('_')[6]).astype(int)
        _app['end_frame'] = _app['segment_name'].apply(lambda f: f.split('_')[7]).astype(int)
        _app['status'] = np.nan
        _app['notes'] = np.nan
        _app['timestamp'] = np.nan
        merged = _app.merge(_df, on=['basename', 'start_frame', 'end_frame'], how='inner', suffixes=('', '_duplicate'))
        common_indices = merged.index
        df = _app.drop(common_indices)
        df = pd.concat([_df, df])
        df = df.sort_values(by=['start_frame', 'basename']).reset_index(drop=True)
        df['video_path'] = df['segment_name'].apply(lambda s: osp.join(self.videos_dir, f'{s}.mp4'))
        return df