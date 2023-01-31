import argparse
from datetime import datetime
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import cv2
import pandas as pd
import numpy as np
import itertools as it
import os

from os import path as osp

from skeleton_tools.skeleton_visualization.mmpose_visualizer import MMPoseVisualizer
from skeleton_tools.utils.constants import REAL_DATA_MOVEMENTS, COLORS, NET_NAME
from skeleton_tools.utils.evaluation_utils import collect_labels, get_intersection

from skeleton_tools.utils.tools import get_video_properties, read_json, read_pkl

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT, COCO_LAYOUT
from sklearn.metrics import confusion_matrix


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
    def __init__(self, df, skeleton_layout, debug=False, qa=False):
        Path('resources/').mkdir(parents=True, exist_ok=True)
        self.out_path = 'resources/qa.csv'
        self.qa = qa
        self.df = df
        self.status_col = 'status_qa2' if self.qa else 'status'
        self.notes_col = 'notes_qa2' if self.qa else 'notes'
        self.timestep_col = 'qa2_timestep' if self.qa else 'qa_timestep'

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
        n, m = self.df[self.df[self.status_col] != Status.NONE.value].shape[0], self.df.shape[0]
        while i < length:
            ret, frame = cap.read()
            if not ret:
                break
            skel_frame = sv._draw_skeletons(np.zeros_like(frame), kp[:, i], kps[:, i], child_id=cids[i])
            frame = np.concatenate((frame, skel_frame), axis=1)
            cv2.putText(frame, f'{n}/{m}',
                        (int(width * 0.05), int(height * 0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (100, 30, 255),
                        2)
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
        def get_vids():
            return list(self.df[(self.df['status'] != Status.NONE.value) & (self.df['status_qa2'] == Status.NONE.value)]['video'].unique()) if self.qa \
                else list(self.df[self.df['status'] == Status.NONE.value]['video'].unique())
        vids = get_vids()
        while any(vids):
            v = vids[0]
            g = self.df[self.df['video'] == v]
            video_path = g.iloc[0]['video_path']
            skeleton = read_pkl(g.iloc[0]['skeleton_path'])
            resolution, fps, frame_count, length = get_video_properties(video_path)
            while not g.empty:
                g = self.df[(self.df['video'] == v) & (self.df[self.status_col] == Status.NONE.value)]
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
                    print(f'Validating: {v} ({s}-{t})' + f' - ({row["annotator"]}, {row["movement"]}, [{s}, {t}], [{row["start_frame"]}, {row["end_frame"]}])' if self.debug else '')
                    frames = self.gen_video(video_path, skeleton, s, t)
                    task = self.executor.submit(lambda: self.validate())
                    self.play(v, frames, done=task.done)
                    status, notes = task.result()
                    if status == Status.STEREOTYPICAL and row['movement'] == 'NoAction':
                        r1, r2, r3 = row.copy(), row.copy(), row.copy()
                        r1[['movement', 'annotator', 'source', 'start_frame', 'end_frame', 'start_time', 'end_time', 'status']] = 'NoAction', 'Human', 'JORDI', row['start_frame'], s, row['start_time'], s / fps, Status.NO_ACTION
                        r2[['movement', 'annotator', 'source', 'start_frame', 'end_frame', 'start_time', 'end_time', 'status']] = 'Stereotypical', 'Human', 'JORDI', s, t, s / fps, t / fps, Status.STEREOTYPICAL
                        r3[['movement', 'annotator', 'source', 'start_frame', 'end_frame', 'start_time', 'end_time', 'status']] = 'NoAction', 'Human', 'JORDI', t, row['end_frame'], t / fps, row['end_time'], Status.NO_ACTION
                        self.df.loc[df.shape[-1]] = r1
                        self.df.loc[df.shape[-1]] = r2
                        self.df.loc[df.shape[-1]] = r3
                        self.df.loc[idx, [self.status_col, self.notes_col, self.timestep_col]] = [Status.SKIP, notes, datetime.now()]
                    else:
                        self.df.loc[idx, [self.status_col, self.notes_col, self.timestep_col]] = [status, notes, datetime.now()]
                    self.df.to_csv(self.out_path, index=False)
                    self.stack.append(idx)
                    g = g.drop(idx)
                except GlobalCommandEvent as gce:
                    self.globals[gce.method].func_ref(*gce.args)
            vids = get_vids()


class GlobalCommandEvent(Exception):
    def __init__(self, method, *args):
        self.method = method
        self.args = args

    def __str__(self):
        return f'{self.method} {self.args}'


def conclusions():
    df = pd.read_csv(r'resources/qa.csv')
    print(f'Number of samples: {len(df)}')
    df = df[~df['movement'].isna() & (df['status'] != "NONE")]
    print(f'Number of qa samples: {len(df)}')
    df['movement_bin'] = df['movement'].apply(lambda m: 0 if 'NoAction' in m else 1)
    df['post_qa'] = df['status'].apply(lambda s: 1 if 'STEREOTYPICAL' in s else 0)

    # df = df[df['source'] == 'JORDI']
    # df = df[df['movement_bin'] == 1]

    ann = pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\combined.csv')
    ann_intervals = {v: intervals for v, intervals in ann.groupby('video')[['start_frame', 'end_frame']]}
    def match_row(row):
        v = row['video']
        intervals = ann_intervals[v].values
        s, t = row['start_frame'], row['end_frame']
        intersections = any([x for x in ((get_intersection((s, t), interval), interval) for interval in intervals) if x[0] is not None])
        return int(intersections)

    df['pre_qa'] = df.apply(match_row, axis=1)
    df.to_csv('resources/qa_processed.csv', index=False)
    # df = df[df['source'] == 'JORDI']
    def calc_results(col):
        tp, fp, tn, fn = df[(df['movement_bin'] == 1) & (df[col] == 1)].shape[0], \
                         df[(df['movement_bin'] == 1) & (df[col] == 0)].shape[0], \
                         df[(df['movement_bin'] == 0) & (df[col] == 0)].shape[0], \
                         df[(df['movement_bin'] == 0) & (df[col] == 1)].shape[0]
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        cm = np.array([[tp, fp], [fn, tn]])
        print(f'Confusion Matrix for {col}: \n{cm}')
        print(f'Results for {col}: Precision - {p}, Recall - {r}, F1 - {f1}')
    for col in ['pre_qa', 'post_qa']:
        calc_results(col)
    print(df.shape)

def load_dataframe(qa_file):
    root = r'Z:\Users\TalBarami\jordi_cross_validation'
    models = ['cv0.pth', 'cv1.pth', 'cv2.pth', 'cv3.pth', 'cv4.pth']
    dfs = [collect_labels(root, osp.join('jordi', m)) for m in models]
    df = pd.concat(dfs, ignore_index=True)
    control = set(pd.read_excel(r'Z:\Users\TalBarami\jordi_cross_validation\control_list.xlsx')['Control children'].tolist())
    df = df[~df['child_id'].isin(control)]
    df['skeleton_path'] = df['video'].apply(lambda v: osp.join(root, v, 'jordi', f'{v}.pkl'))
    df['movement'] = df['movement'].fillna('Stereotypical')
    df = df.groupby(['video', 'start_frame', 'end_frame']).agg({'movement': lambda l: ','.join(set(it.chain(*[[m.strip() for m in x.split(',')] for x in l]))), **{c: lambda x: x.iloc[0] for c in df.columns if c != 'movement'}}).reset_index(drop=True)
    df['status'] = Status.NONE.value
    df['notes'] = ''
    df['qa_timestep'] = None
    if osp.exists(qa_file):
        qa = pd.read_csv(qa_file)
        for i, row in qa.iterrows():
            df_row = df.loc[(df['video'] == row['video']) & (df['start_frame'] == row['start_frame']) & (df['end_frame'] == row['end_frame'])]
            if len(df_row) == 1:
                df.loc[df_row.index, ['status', 'notes', 'qa_timestep']] = row['status'], row['notes'], row['qa_timestep']
            elif len(df_row) > 1:
                movements = ','.join(df_row['movement'].unique())
                validated = df_row[df_row['status'] != Status.NONE.value]
                df_row = df_row.iloc[0]
                df_row['movement'] = movements
                if not validated.empty:
                    val = validated.iloc[0]
                    df_row['status'] = val['status']
                    df_row['notes'] = val['notes']
                df.loc[df_row.name] = df_row
            else:
                print(f'Error: QA row {i} does not match any row in the original dataframe.')
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-c', '--conclude', action='store_true')
    parser.add_argument('-r', '--reload', action='store_true')
    parser.add_argument('--qa', action='store_true')
    args = parser.parse_args()
    if args.conclude:
        conclusions()
    else:
        qa_file = 'resources/qa.csv'
        if args.reload:
            df = load_dataframe(qa_file)
        else:
            df = pd.read_csv(qa_file)
        if 'status_qa2' not in df.columns:
            df['status_qa2'] = Status.NONE.value
            df['notes_qa2'] = ''
            df['qa2_timestep'] = None
        ann = Reannotator(df, COCO_LAYOUT, debug=args.debug, qa=args.qa)
        ann.run()
