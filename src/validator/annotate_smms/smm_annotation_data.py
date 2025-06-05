from datetime import datetime
from os import path as osp

import pandas as pd

from validator.data_handler import DataHandler


class SMMsAnnotationData(DataHandler):
    def __init__(self, annotator_id, test_dir, filename, qa=None):
        self.test_dir = test_dir
        self.videos_dir = osp.join(self.test_dir, 'videos')
        self.annotations_dir = osp.join(self.test_dir, 'annotations')
        self.base_template = osp.join(self.test_dir, filename)
        self.i = 0
        self.qa = qa
        if self.qa is not None:
            annotator_id = f'{annotator_id}_qa'
        super().__init__(annotator_id=annotator_id, annotations_file=osp.join(self.annotations_dir, f'{annotator_id}.csv'))

    def get_indices(self):
        if self.qa is not None:
            def idxs(df):
                y_true = df['status'] == 'SMM'
                y_pred = df['conf_smm'] > 0.7
                idx = df[~df['status'].isna() & (y_true != y_pred)].index
                return idx

            df1 = pd.read_csv(osp.join(self.annotations_dir, f'{self.qa[0]}.csv'))
            df2 = pd.read_csv(osp.join(self.annotations_dir, f'{self.qa[1]}.csv'))
            idx = idxs(df1).union(idxs(df2))
            return idx
        return super().get_indices()

    def add(self, idx, ann):
        status, notes, smm_type, video = ann['status'], ann['notes'], ann['smm_type'], ann['video']
        self.stack.append(idx)
        self.df.loc[idx, ['status', 'smm_type', 'notes', 'video', 'timestamp', 'annotator']] = [status, str(smm_type), notes, video, pd.Timestamp(datetime.now()), self.annotator_id]
        self.save()

    def collect_annotations(self):
        if osp.exists(self.annotations_file):
            df = pd.read_csv(self.annotations_file)
            if 'video' not in df.columns:
                df['video'] = None
        else:
            df = pd.read_csv(self.base_template)
            df[['status', 'smm_type', 'notes', 'timestamp', 'annotator', 'video']] = [None, None, None, None, None, None]
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df