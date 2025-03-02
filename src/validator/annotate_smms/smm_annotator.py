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
    def __init__(self, root, annotator_id, filename, debug=False):
        self.filename = filename
        self.am = ANCANManager()
        self.smm_types = sorted(['Hand flapping', 'Tapping', 'Clapping', 'Fingers', 'Body rocking',
                          'Tremor', 'Spinning in circle', 'Toe walking', 'Back and forth',
                          'Head movement', 'Playing with object', 'Jumping in place', 'Legs movement', 'Feeling texture', 'Other'])

        self.debug = debug
        super().__init__(root, annotator_id)

    def init_status_types(self):
        return ['SMM', 'No Action', 'Corrupted Video', 'Skip']

    def init_data_handler(self):
        return SMMsAnnotationData(test_dir=self.root, annotator_id=self.annotator_id, filename=self.filename)

    def init_video_player(self):
        return AssessmentVideoPlayer(osp.join(RESOURCES_ROOT, 'config.json'))
        # return VideoPlayer(osp.join(RESOURCES_ROOT, 'config.json'))

    def validate(self, opts, score, cam_ids, fps):
        ext = f' (score: {score})' if self.debug else ''
        ans = self.choose(f'Choose status:{ext}', opts)
        status = opts[ans[0]]
        smm_type = self.choose('Choose SMM type(s):', self.smm_types, offset=1) if status == 'SMM' else []
        cameras = self.choose('Choose camera(s):', cam_ids, as_is=True) if status == 'SMM' else []
        notes = input('Save notes: ') if status == 'Skip' else None
        return {'status': status, 'notes': notes, 'smm_type': [self.smm_types[x] for x in smm_type], 'cameras': [cam_ids[x] for x in cameras], 'fps': fps}

    def add_to_queue(self, row):
        # name, start, end = row['assessment'], row['start'], row['end']
        name, start, end = row['assessment'], row['start'], row['end']
        df = self.am.for_assessment(name)
        fps = df['fps'].mode()[0]
        basenames = df[df['fps'] == fps]['basename']
        filenames = [osp.join(self.data_handler.videos_dir, f'{b}_{start}_{end}.mp4') for b in basenames]
        filenames = [f for f in filenames if osp.exists(f)]
        if not filenames:
            return row.name, None, None, None
        frames, fps = self.video_player.gen_video(filenames)
        # filename = osp.join(self.data_handler.videos_dir, f'{name}_{start}_{end}.mp4')
        # if not osp.exists(filename):
        #     return row.name, None, None
        # frames, fps = self.video_player.gen_video(filename)
        return row.name, frames, fps, {'score': row['conf_smm'], 'cam_ids': [osp.basename(f).split('_')[-3] for f in filenames], 'fps': fps}

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
    parser.add_argument('--root', type=str, help='Root directory')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    root = osp.join(USERS_ROOT, 'TalBarami', 'smm_project', 'manual_annotations')
    annotators = ['Liora', 'Noa', 'Shaked']
    annotator = select_annotator(annotators)
    print('Starting annotations tool...')
    program = SMMAnnotator(root, annotator, filename='annotations_test14.csv', debug=args.debug)
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