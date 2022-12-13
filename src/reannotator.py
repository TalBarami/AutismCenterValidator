import argparse
from datetime import datetime
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import cv2
import pandas as pd
import numpy as np
import os

from os import path as osp

from skeleton_tools.skeleton_visualization.numpy_visualizer import MMPoseVisualizer
from skeleton_tools.utils.constants import REAL_DATA_MOVEMENTS, COLORS
from skeleton_tools.utils.evaluation_utils import collect_labels

from skeleton_tools.utils.tools import get_video_properties, read_json, read_pkl

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT, COCO_LAYOUT


class Status(Enum):
    NONE = 'NONE'
    STEREOTYPICAL = 'Stereotypical'
    NO_ACTION = 'NoAction'
    SKIP = 'SKIP'

class Resolution(Enum):
    AUTO = 'AUTO'
    MANUAL = 'MANUAL'

class Global:
    def __init__(self, name, args_count, func_ref):
        self.name = name
        self.args_count = args_count
        self.func_ref = func_ref


class Reannotator:
    def __init__(self, df, skeleton_layout, debug=False):
        Path('resources/').mkdir(parents=True, exist_ok=True)
        self._df = df
        self.out_path = 'resources/qa.csv'
        if osp.exists(self.out_path):
            self.df = pd.read_csv(self.out_path)
        else:
            self.df = self._df.copy()
            self.df['status'] = Status.NONE.value
            self.df['notes'] = ''
        self.executor = ThreadPoolExecutor()
        self.skeleton_layout = skeleton_layout
        self.sequence_length = 250
        self.stack = []
        self.debug = debug

        self.speed = 3
        self.globals = {g.name: g for g in [Global('revert', 0, self.revert), Global('speed', 1, self.set_speed), Global('resolution', 2, self.set_resolution), Global('reset', 0, self.reset)]}
        self.resolution_method = Resolution.AUTO
        self.resolution = (0, 0)

    def revert(self):
        if len(self.stack) == 0:
            return
        self.df.loc[self.stack.pop(), ['status', 'notes']] = Status.NONE.value, ''
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

    def exit(self):
        self.df.to_csv(self.out_path, index=False)
        exit(0)

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

    def get_resolution(self, resolution):
        return self.resolution if self.resolution_method == Resolution.MANUAL else resolution

    def play(self, video_name, frames, done=None):
        fps = 30
        org_resolution = frames[0].shape[:2]
        i = 0
        while True:
            frame = frames[i]
            resolution = self.get_resolution(org_resolution)
            if resolution != org_resolution:
                frame = cv2.resize(frame, resolution)

            if self.speed <= 4:
                sleep(1 / (fps * np.power(2, self.speed)))
            cv2.imshow(video_name, frame)
            i += 1
            if i >= len(frames):
                i = 0
            if (done and done()) or (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    def validate(self):
        opts = [Status.STEREOTYPICAL, Status.NO_ACTION, Status.SKIP]
        ans = self.choose('Choose status:', [s. value for s in opts])
        status = opts[ans[0]]
        result_notes = input('Save notes: ') if status == Status.SKIP else None
        return status, result_notes

    def gen_video(self, video_path, skeleton, s, t):
        sv = MMPoseVisualizer(self.skeleton_layout)
        cap = cv2.VideoCapture(video_path)
        fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width *= 2
        length = t - s
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        kp = skeleton['keypoint'][:, s:t]
        kps = skeleton['keypoint_score'][:, s:t]
        cids = skeleton['child_ids'][s:t]
        i = 0
        frames = []
        while i < length:
            ret, frame = cap.read()
            skel_frame = sv.draw_skeletons(np.zeros_like(frame), kp[:, i], kps[:, i], child_id=cids[i])
            frame = np.concatenate((frame, skel_frame), axis=1)
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
            frames.append(frame)
            i += 1
        cap.release()
        return frames

    def run(self):
        vids = list(self.df[self.df['status'] == Status.NONE.value]['video'].unique())
        while any(vids):
            v = vids[0]
            g = self.df[self.df['video'] == v]
            video_path = g.iloc[0]['video_path']
            skeleton = read_pkl(g.iloc[0]['skeleton_path'])
            while not g.empty:
                g = self.df[(self.df['video'] == v) & (self.df['status'] == Status.NONE.value)]
                row, idx = g.iloc[0], g.index[0]
                try:
                    s, t = row['start_frame'], row['end_frame']
                    if row['movement'] == 'NoAction':
                        total_length = t - s
                        l = int(np.random.normal(self.sequence_length, self.sequence_length / 8))
                        if total_length > l:
                            s = np.random.randint(0, total_length - l)
                            t = s + l
                    s, t = int(s), int(t)
                    print(f'Validating: {v} ({s}-{t})' + f' - {row["movement"]}' if self.debug else '')
                    frames = self.gen_video(video_path, skeleton, s, t)
                    task = self.executor.submit(lambda: self.validate())
                    self.play(v, frames, done=task.done)
                    status, notes = task.result()
                    self.df.loc[idx, ['status', 'notes']] = [status, notes]
                    self.df.to_csv(self.out_path, index=False)
                    self.stack.append(idx)
                    g = g.drop(idx)
                except GlobalCommandEvent as gce:
                    self.globals[gce.method].func_ref(*gce.args)
            vids = list(self.df[self.df['status'] == Status.NONE.value]['video'].unique())


class GlobalCommandEvent(Exception):
    def __init__(self, method, *args):
        self.method = method
        self.args = args

    def __str__(self):
        return f'{self.method} {self.args}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()

    root = r'Z:\Users\TalBarami\jordi_cross_validation'
    models = ['cv0.pth', 'cv1.pth']
    dfs = [collect_labels(root, osp.join('jordi', m)) for m in models]
    df = pd.concat(dfs, ignore_index=True)
    df['skeleton_path'] = df['video'].apply(lambda v: osp.join(root, v, 'jordi', f'{v}.pkl'))

    ann = Reannotator(df, COCO_LAYOUT, debug=args.debug)
    ann.run()
