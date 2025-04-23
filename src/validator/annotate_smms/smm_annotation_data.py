from datetime import datetime
from os import path as osp

import pandas as pd

from validator.data_handler import DataHandler


class SMMsAnnotationData(DataHandler):
    def __init__(self, annotator_id, test_dir, filename):
        self.test_dir = test_dir
        self.videos_dir = osp.join(self.test_dir, 'videos')
        self.annotations_dir = osp.join(self.test_dir, 'annotations')
        self.base_template = osp.join(self.test_dir, filename)
        self.i = 0
        super().__init__(annotator_id=annotator_id, annotations_file=osp.join(self.annotations_dir, f'{annotator_id}.csv'))

    # def get_indices(self):
    #     df = self.df[self.df['status'].isna()]
    #     high = df[df['group'] == 'high'].index
    #     med = df[df['group'] == 'med'].sample(frac=0.2).index
    #     low = df[df['group'] == 'low'].sample(frac=0.1).index
    #     return high.union(med).union(low)

    def add(self, idx, ann):
        status, notes, smm_type = ann['status'], ann['notes'], ann['smm_type']
        self.stack.append(idx)
        self.df.loc[idx, ['status', 'smm_type', 'notes', 'timestamp', 'annotator']] = [status, str(smm_type), notes, pd.Timestamp(datetime.now()), self.annotator_id]
        self.save()

    def collect_annotations(self):
        if osp.exists(self.annotations_file):
            df = pd.read_csv(self.annotations_file)
        else:
            df = pd.read_csv(self.base_template)
            df[['status', 'smm_type', 'notes', 'timestamp', 'annotator']] = [None, None, None, None, None]
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df