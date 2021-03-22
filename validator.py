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

from utils import draw_json_skeletons, Status, REAL_DATA_MOVEMENTS, COLORS, BODY_25_LAYOUT, read_json, get_video_properties

class Global:
    def __init__(self, name, args_count, func_ref):
        self.name = name
        self.args_count = args_count
        self.func_ref = func_ref

class Validator:

    def __init__(self, files, skeleton_layout):
        Path('resources/').mkdir(parents=True, exist_ok=True)
        self.files = files
        self.out_path = 'resources/qa.csv'
        self.df = pd.read_csv(self.out_path) if path.isfile(self.out_path) else pd.DataFrame(columns=['video_name', 'segment_name', 'start_time', 'end_time', 'start_frame', 'end_frame', 'status', 'action', 'child_ids', 'time', 'notes'])
        self.executor = ThreadPoolExecutor()
        self.skeleton_layout = skeleton_layout
        self.idx = 0
        self.speed = 1
        self.globals = {g.name: g for g in [Global('revert', 0, self.revert), Global('speed', 1, self.set_speed)]}

    def revert(self):
        if self.idx > 1:
            self.idx -= 1
            self.df.drop(self.df.tail(1).index, inplace=True)
            self.df.to_csv(self.out_path, index=False)

    def set_speed(self, i):
        try:
            i = float(i)
            if i < 0.5:
                return
            self.speed = i
        except ValueError:
            pass

    def valid_global(self, str):
        if not str.startswith('-') or len(str) < 2:
            return False
        cmd = str[1:].split(' ')
        return any(g for g in self.globals.values() if len(cmd) == g.args_count+1 and cmd[0] == g.name)

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

    def play(self, video_path, skeleton=None, label=None, done=None):
        cap = cv2.VideoCapture(video_path)
        width, height, fps, length = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        i = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if skeleton is not None and i < len(skeleton):
                    draw_json_skeletons(frame, skeleton[i]['skeleton'], (width, height), self.skeleton_layout, normalized=True)
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
                if label is not None:
                    cv2.putText(frame, label,
                                (int(width * 0.4), int(height * 0.1)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 255, 0),
                                5)
                sleep(1 / (np.power(2, self.speed) * fps))
                cv2.imshow('skeleton', frame)
                i += 1
            else:
                i = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            if (done and done()) or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        cap.release()

    def validate(self, base_label, max_id):
        result_labels = [base_label]
        result_cids = [-1]
        result_notes = ''

        opts = [s for s in [Status.OK, Status.NO_ACTION_OBSERVED, Status.NO_SKELETON, Status.WRONG_LABEL, Status.NO_SKELETON_WRONG_LABEL, Status.SKIP]]
        ans = self.choose('Choose status:', [s.value for s in opts])
        status = opts[ans[0]]
        if status == Status.SKIP:
            result_notes = input('Save notes: ')
        if status in [Status.WRONG_LABEL, Status.NO_SKELETON_WRONG_LABEL]:
            ans = self.choose('Choose label(s):', REAL_DATA_MOVEMENTS)
            result_labels = [REAL_DATA_MOVEMENTS[i] for i in ans]
            status = Status.OK if status == Status.WRONG_LABEL else Status.NO_SKELETON_WRONG_LABEL
        if status != Status.SKIP and status != Status.NO_ACTION_OBSERVED and 'Other' in result_labels:
            result_notes = input('Save notes: ')
        if status == Status.OK:
            result_cids = self.choose('Choose child id(s):', [f'({COLORS[i % len(COLORS)]["name"]})' for i in range(max_id + 1)], offset=0)

        return status, result_labels, result_cids, result_notes

    def run(self):
        tagged_data = set(self.df['segment_name'].unique())
        while True:
            try:
                if len(tagged_data) == len(self.files):
                    break
                name, vpath, spath = self.files[self.idx]
                self.idx += 1

                _df = self.df[self.df['segment_name'] == name]
                if not _df.empty:  # and if the tagged row is not None / Skip / Whatever I decide
                    continue

                skeleton = read_json(spath)['data'] if spath else None
                max_id = np.max([s['person_id'] for v in skeleton for s in v['skeleton']]) if spath else 0
                split = name.split('_')
                basename = '_'.join(split[:-3])
                _, fps, _ = get_video_properties(vpath)
                base_label, start_frame, end_frame = split[-3:]
                start_frame, end_frame = int(start_frame), int(end_frame)
                start_time, end_time = int(start_frame / fps), int(end_frame / fps)

                print(f'Validating: {vpath}')
                task = self.executor.submit(lambda: self.validate(base_label, max_id))
                self.play(vpath, skeleton=skeleton, label=base_label, done=task.done)
                status, labels, cids, notes = task.result()
                for label in labels:
                    self.df.loc[self.df.shape[0]] = [basename, name, start_time, end_time, start_frame, end_frame, status, label, cids, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), notes]
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


# global_cmds = {
#     '-break': True,
#     '-back': True,
#     '-speed': True,
#     '-label': True,
# }
if __name__ == '__main__':
    videos_path = 'D:/segmented_videos'
    skeletons_path = 'D:/skeletons/data'
    # files = [(path.splitext(name)[0], path.join(videos_path, name), path.join(skeletons_path, f'{path.splitext(name)[0]}.json')) for name in os.listdir(videos_path) if
    #               path.isfile(path.join(skeletons_path, f'{path.splitext(name)[0]}.json'))]
    bad = read_json('C:/Users/owner/PycharmProjects/RepetitiveMotionRecognition/resources/qa/bad_files.json')
    files = [(path.splitext(name)[0], path.join(videos_path, name), None) for name in os.listdir(videos_path) if path.splitext(name)[0] in bad]
    val = Validator(files, BODY_25_LAYOUT)
    val.run()
