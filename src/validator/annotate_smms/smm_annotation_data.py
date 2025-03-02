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

    def add(self, idx, ann):
        status, notes, smm_type, cam_ids, fps = ann['status'], ann['notes'], ann['smm_type'], ann['cameras'], ann['fps']
        self.stack.append(idx)
        self.df.loc[idx, ['status', 'smm_type', 'cameras', 'notes', 'timestep', 'annotator']] = [status, str(smm_type), str(cam_ids), notes, pd.Timestamp(datetime.now()), self.annotator_id]
        self.save()

    def collect_annotations(self):
        if osp.exists(self.annotations_file):
            df = pd.read_csv(self.annotations_file)
        else:
            df = pd.read_csv(self.base_template)[['assessment', 'conf_smm', 'group', 'start', 'end']]
            df[['status', 'smm_type', 'cameras', 'notes', 'timestep', 'annotator']] = [None, None, None, None, None, None]
        df = df.sample(frac=1).reset_index(drop=True)
        return df