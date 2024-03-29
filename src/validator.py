from datetime import datetime
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import cv2
import pandas as pd
import numpy as np
import os

from os import path

from skeleton_tools.utils.constants import REAL_DATA_MOVEMENTS, COLORS

from skeleton_tools.utils.tools import get_video_properties, read_json, read_pkl

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT, COCO_LAYOUT


class Status(Enum):
    NONE = 'NONE'
    OK = 'OK'
    NO_ACTION_OBSERVED = 'NO_ACTION_OBSERVED'
    NO_SKELETON = 'NO_SKELETON'
    WRONG_LABEL = 'WRONG_LABEL'
    NO_SKELETON_WRONG_LABEL = 'NO_SKELETON + WRONG_LABEL'
    SKIP = 'SKIP'

class Resolution(Enum):
    AUTO = 'AUTO'
    MANUAL = 'MANUAL'

class Global:
    def __init__(self, name, args_count, func_ref):
        self.name = name
        self.args_count = args_count
        self.func_ref = func_ref


class Validator:
    def __init__(self, files, skeleton_layout):
        Path('resources/').mkdir(parents=True, exist_ok=True)
        self.files = files
        self.out_path = 'resources/qa_backup.csv'
        self.df = pd.read_csv(self.out_path) if path.isfile(self.out_path) else pd.DataFrame(columns=['video_name', 'segment_name', 'start_time', 'end_time', 'start_frame', 'end_frame', 'status', 'actions', 'child_ids', 'time', 'notes'])
        self.executor = ThreadPoolExecutor()
        self.skeleton_layout = skeleton_layout
        self.idx = 0
        self.n = len(files)
        self.speed = 3
        self.globals = {g.name: g for g in [Global('revert', 0, self.revert), Global('speed', 1, self.set_speed), Global('resolution', 2, self.set_resolution), Global('reset', 0, self.reset)]}
        self.resolution_method = Resolution.AUTO
        self.resolution = (0, 0)

    def revert(self):
        if self.idx > 1:
            self.idx -= 1
            self.df.drop(self.df.tail(1).index, inplace=True)
            self.df.to_csv(self.out_path, index=False)

    def reset(self):
        return

    def set_speed(self, i):
        try:
            i = float(i)
            if i < 0.5:
                return
            self.speed = i
        except ValueError:
            pass

    def set_resolution(self, width, height):
        self.resolution = int(width), int(height)
        self.resolution_method = Resolution.MANUAL if np.sum(self.resolution) > 0 else Resolution.AUTO

    def valid_global(self, str):
        if not str.startswith('-') or len(str) < 2:
            return False
        cmd = str[1:].split(' ')
        return any(g for g in self.globals.values() if len(cmd) == g.args_count + 1 and cmd[0] == g.name)

    def choose(self, msg, lst, offset=1):
        while True:
            print(msg)
            for i, e in enumerate(lst):
                print(f'{i + offset}. {e}')
            result = input()
            if self.valid_global(result):
                raise GlobalCommandEvent(*result[1:].split(' '))
            result = [s for s in result.split(' ') if s]
            if len(result) > 0 and all(s.isdigit() and (offset <= int(s) < len(lst) + offset) for s in result):
                result = [int(s) - offset for s in result]
                return [i for i in result]
            print('Error: Wrong selection.')

    def get_resolution(self, cap):
        return self.resolution if self.resolution_method == Resolution.MANUAL else (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))

    def play(self, video_path, labels=None, done=None):
        cap = cv2.VideoCapture(video_path)
        fps, length = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        i = 0
        while True:
            width, height = self.get_resolution(cap)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))
                if i < 10:
                    cv2.putText(frame, 'Reset',
                                (int(width * 0.5), int(height * 0.5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3,
                                (0, 0, 255),
                                5)
                cv2.putText(frame, f'{i}/{length}',
                            (int(width * 0.05), int(height * 0.95)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (100, 30, 255),
                            2)
                cv2.putText(frame, f'{self.idx}/{self.n}',
                            (int(width * 0.05), int(height * 0.05)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (100, 30, 255),
                            2)
                if labels is not None:
                    cv2.putText(frame, ','.join(labels),
                                (int(width * 0.2), int(height * 0.1)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 255, 0),
                                5)
                if self.speed <= 4:
                    sleep(1 / (np.power(2, self.speed) * fps))
                cv2.imshow('skeleton', frame)
                i += 1
            else:
                i = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            if (done and done()) or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        cap.release()

    def validate(self, org_labels, max_id):
        result_labels = org_labels
        result_cids = [-1]
        result_notes = ''

        opts = [s for s in [Status.OK, Status.NO_ACTION_OBSERVED, Status.NO_SKELETON, Status.WRONG_LABEL, Status.NO_SKELETON_WRONG_LABEL, Status.SKIP]]
        ans = self.choose('Choose status:', [s.value for s in opts])
        status = opts[ans[0]]
        if status == Status.SKIP:
            result_notes = input('Save notes: ')
        if status in [Status.WRONG_LABEL, Status.NO_SKELETON_WRONG_LABEL]:
            ans = self.choose('Choose label(s):', REAL_DATA_MOVEMENTS[:-1])
            result_labels = [REAL_DATA_MOVEMENTS[i] for i in ans]
            status = Status.OK if status == Status.WRONG_LABEL else Status.NO_SKELETON_WRONG_LABEL
        if status != Status.SKIP and status != Status.NO_ACTION_OBSERVED and 'Other' in result_labels:
            result_notes = input('Save notes: ')
        # if status == Status.OK:
        #     result_cids = self.choose('Choose child id(s):', [f'({COLORS[i % len(COLORS)]["name"]})' for i in range(max_id + 1)], offset=0)

        return status, result_labels, result_cids, result_notes

    def run(self):
        tagged_data = set(self.df['segment_name'].unique())
        while True:
            try:
                if set([f[0] for f in self.files]).issubset(tagged_data):
                    print(f'Finishd tagging {self.idx} files')
                    break
                name, vpath, spath = self.files[self.idx]
                self.idx += 1

                _df = self.df[self.df['segment_name'] == name]
                if not _df.empty:  # and if the tagged row is not None / Skip / Whatever I decide
                    continue

                # skeleton = read_pkl(spath)
                # max_id = np.max([s['person_id'] for v in skeleton for s in v['skeleton']]) if spath and any([v['skeleton'] for v in skeleton]) else 0
                max_id = 0 # TODO: FIX
                split = name.split('_')
                basename = '_'.join(split[:-3])
                _, fps, _, _ = get_video_properties(vpath)
                org_labels, start_frame, end_frame = split[-3:]
                org_labels = org_labels.split(',')
                start_frame, end_frame = int(start_frame), int(end_frame)
                start_time, end_time = int(start_frame / fps), int(end_frame / fps)

                print(f'Validating: {vpath}')
                task = self.executor.submit(lambda: self.validate(org_labels, max_id))
                self.play(vpath, labels=org_labels, done=task.done)
                status, labels, cids, notes = task.result()
                self.df.loc[self.df.shape[0]] = [basename, name, start_time, end_time, start_frame, end_frame, status, labels, cids, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notes]
                self.df.to_csv(self.out_path, index=False)
            except GlobalCommandEvent as g:
                self.globals[g.method].func_ref(*g.args)
                self.idx -= 1


class GlobalCommandEvent(Exception):
    def __init__(self, method, *args):
        self.method = method
        self.args = args

    def __str__(self):
        return f'{self.method} {self.args}'


if __name__ == '__main__':
    root = r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\movement_type_qa'
    videos_path = path.join(root, 'skeleton_videos')
    skeletons_path = path.join(root, 'skeletons', 'raw')
    # labels = read_json('D:/skeletons/label.json')
    # _actions = {'Hand flapping', 'Tapping', 'Clapping', 'Body rocking', 'Tremor', 'Head movement'}
    # files = [(path.splitext(name)[0], path.join(videos_path, name), path.join(skeletons_path, f'{path.splitext(name)[0]}.json')) for name in os.listdir(videos_path) if
    #               path.isfile(path.join(skeletons_path, f'{path.splitext(name)[0]}.json'))
    #          and (name.startswith('2') or name.startswith('9') or name.startswith('6'))
    #          and labels[path.splitext(name)[0]]['label'] in _actions]
    # bad = read_json('C:/Users/owner/PycharmProjects/RepetitiveMotionRecognition/resources/qa/bad_files.json')
    # files = [(path.splitext(name)[0], path.join(videos_path, name), None) for name in os.listdir(videos_path) if path.splitext(name)[0] in bad]
    files = [(path.splitext(name)[0], path.join(videos_path, name), path.join(skeletons_path, f'{path.splitext(name)[0]}.pkl')) for name in os.listdir(videos_path) if
             path.isfile(path.join(skeletons_path, f'{path.splitext(name)[0]}.pkl'))]
    val = Validator(files, COCO_LAYOUT)
    val.run()
