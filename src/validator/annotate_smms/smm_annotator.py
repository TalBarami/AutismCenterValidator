import os
from os import path as osp
import argparse

import numpy as np
from asdhub.ancan_db.ancan_db import ANCANManager
from asdhub.constants import USERS_ROOT

from validator.annotate_smms.smm_annotation_data import SMMsAnnotationData
from validator.annotator import Annotator
from validator.constants import RESOURCES_ROOT
from validator.video_player import VideoPlayer, AssessmentVideoPlayer


class SMMAnnotator(Annotator):
    def __init__(self, root, annotator_id, filename, debug=False, qa=False, ann_type=True):
        self.filename = filename
        self.am = ANCANManager()
        self.smm_types = sorted(['Hand flapping', 'Tapping', 'Clapping', 'Fingers', 'Body rocking',
                          'Tremor', 'Spinning in circle', 'Toe walking', 'Back and forth',
                          'Head movement', 'Playing with object', 'Jumping in place', 'Legs movement', 'Feeling texture', 'Other'])

        self.debug = debug
        self.qa = qa
        self.ann_type = ann_type
        super().__init__(root, annotator_id)

    def init_status_types(self):
        return ['SMM', 'Repetitive (Non-SMM)', 'No Action', 'Corrupted Video', 'Skip']

    def init_data_handler(self):
        return SMMsAnnotationData(test_dir=self.root, annotator_id=self.annotator_id, filename=self.filename, qa=['Noa', 'Shaked'] if self.qa else None)

    def init_video_player(self):
        return VideoPlayer(osp.join(RESOURCES_ROOT, 'config.json'))

    def validate(self, opts, score):
        ext = f' (score: {score})' if self.debug else ''
        # if self.qa:
        #     ext += f' (Model: {1 if score > 0.7 else 0}, Annotators: {1 if score < 0.7 else 0})'
        ans = self.choose(f'Choose status:{ext}', opts)
        status = opts[ans[0]]
        smm_type = self.choose('Choose SMM type(s):', self.smm_types, offset=1) if (self.ann_type and status == 'SMM') else []
        notes = input('Save notes: ') if status == 'Skip' else None
        return {'status': status, 'notes': notes, 'smm_type': [self.smm_types[x] for x in smm_type]}

    def add_to_queue(self, row):
        name, start, end = row['assessment'], row['start'], row['end']
        df = self.am.for_assessment(name).copy()
        df['cam_id'] = df['basename'].apply(lambda b: int(b.split('_')[-1]))
        if row['group'] == 'low':
            cam_ids = df['cam_id'].unique()
            cam_id = np.random.choice(cam_ids, 1)[0]
        else:
            cam_id = row['cam_id']
        fname = df[df['cam_id'] == cam_id]['basename']
        if fname.empty:
            print(f'No video found for {name} {cam_id}')
            return row.name, None, None, {}
        fname = fname.iloc[0]
        filename = osp.join(self.data_handler.videos_dir, f'{fname}_{start}_{end}.mp4')
        if not osp.exists(filename):
            print(f'Video not found: {filename}')
            return row.name, None, None, {}
        frames, fps = self.video_player.gen_video(filename)
        if len(frames) == 0:
            return row.name, None, None, {}
        return row.name, frames, fps, {'score': row['conf_smm']}

def select_annotator(annotators):
    offset = 1
    while True:
        print(f'Identify:')
        for i, e in enumerate(annotators):
            print(f'{i + offset}. {e}')
        result = input()
        result = [s for s in result.split(' ') if s]
        if len(result) > 0 and all(s.isdigit() and (offset <= int(s) < len(annotators) + offset) for s in result):
            result = [int(s) - offset for s in result][0]
            print(f'You are identifying as: {annotators[result]} - ARE YOU SURE? y/n')
            ans = input()
            if ans == 'y':
                return annotators[result]
            continue
        print('Error: Wrong selection.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotator')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    parser.add_argument('-qa', '--qa', action='store_true', help='QA mode')
    args = parser.parse_args()
    root = osp.join(USERS_ROOT, 'TalBarami', 'smm_project', 'manual_annotations')
    annotators = ['Liora', 'Noa', 'Shaked']
    annotator = select_annotator(annotators)
    print('Starting annotations tool...')
    program = SMMAnnotator(root, annotator, filename='annotations_test14.csv', debug=args.debug, qa=args.qa, ann_type=annotator!='Liora')
    print(f'Welcome {annotator}! Here\'s a Short guide for you:')
    print(f'1. Choose status: {", ".join(program.status_types)}')
    print('2. If you typed \'Skip\', you will be asked to save notes.')
    print('3. If you typed \'SMM\', you will be asked to choose the SMM type. You can choose multiple types, separated by space.')
    print('4. To exit annotator, type \'-exit\'.')
    print('5. To revert last action, type \'-revert\'.')
    print('6. To change video speed, type \'-speed <speed>\'.')
    print('7. To change video resolution, type \'-resolution <width> <height>\'.')
    print('8. To restart the video, type \'-reset\'.')
    program.run()