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
from skeleton_tools.utils.constants import REAL_DATA_MOVEMENTS, COLORS, NET_NAME, DB_PATH
from skeleton_tools.utils.evaluation_utils import collect_labels, get_intersection

from skeleton_tools.utils.tools import get_video_properties, read_json, read_pkl, write_json

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
        self.out_path = r'Z:\Users\TalBarami\videos_qa\qa.csv'
        self.cfg_file = 'resources/config.json'
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

        if osp.exists(self.cfg_file):
            self.cfg = read_json(self.cfg_file)
        else:
            self.cfg = {'speed': 3, 'resolution': (0, 0)}
        self.speed = self.cfg['speed']
        self.resolution = self.cfg['resolution']
        self.globals = {g.name: g for g in [Global('exit', 0, self.exit), Global('revert', 0, self.revert), Global('speed', 1, self.set_speed), Global('resolution', 2, self.set_resolution), Global('reset', 0, self.reset)]}
        self.resolution_method = Resolution.AUTO

        self.set_speed(self.speed)
        self.set_resolution(*self.resolution)

    def save(self):
        self.df.to_csv(self.out_path, index=False)
        self.cfg['speed'] = self.speed
        self.cfg['resolution'] = self.resolution
        write_json(self.cfg, self.cfg_file)

    def exit(self):
        self.save()
        exit(0)

    def revert(self):
        print(f'Unable to revert. Stack is empty.')
        return
        # if len(self.stack) == 0:
        #     return
        # self.df.loc[self.stack.pop(), ['status', 'notes']] = Status.NONE.value, ''
        # self.df.to_csv(self.out_path, index=False)

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
        cv2.destroyAllWindows()

    def validate(self):
        opts = [Status.STEREOTYPICAL, Status.NO_ACTION, Status.SKIP]
        ans = self.choose('Choose status:', [s.value for s in opts])
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
        n = self.df[self.df[self.status_col] != Status.NONE.value].shape[0]
        # m = self.df[(self.df['status'] != Status.NONE.value)].shape[0] if self.qa else self.df.shape[0]
        # (self.df['status'] == Status.NONE.value) & (self.df['movement'] != 'NoAction') & (self.df['source'] != 'Human')
        m = self.df[(self.df['status'] != Status.NONE.value) & (self.df['movement'] != 'NoAction')].shape[0] if self.qa else self.df[(self.df['movement'] != 'NoAction') & (self.df['source'] != 'Human')].shape[0]
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
            return list(self.df[(self.df['status'] != Status.NONE.value) & (self.df['status_qa2'] == Status.NONE.value) & (self.df['movement'] != 'NoAction') & (self.df['source'] != 'Human')]['video'].unique()) if self.qa \
                else list(self.df[(self.df['status'] == Status.NONE.value) & (self.df['movement'] != 'NoAction') & (self.df['source'] != 'Human')]['video'].unique())
        vids = get_vids()
        while any(vids):
            v = vids[0]
            g = self.df[self.df['video'] == v]
            video_path = g.iloc[0]['video_path']
            skeleton = read_pkl(g.iloc[0]['skeleton_path'])
            while not g.empty:
                g = self.df[(self.df['video'] == v) & (self.df[self.status_col] == Status.NONE.value) & (self.df['movement'] != 'NoAction') & (self.df['source'] != 'Human')]
                row, idx = g.iloc[0], g.index[0]
                try:
                    s, t, m = row['start_frame'], row['end_frame'], row['model']
                    if row['movement'] == 'NoAction':
                        total_length = t - s
                        l = int(np.random.normal(self.sequence_length, self.sequence_length / 8))
                        if total_length > l:
                            s = s + np.random.randint(0, total_length - l)
                            t = s + l
                    s, t = int(s), int(t)
                    if t <= s:
                        print(f'Video: {v}, s={s}, t={t}, skip...')
                        self.df.loc[idx, self.status_col] = Status.SKIP
                    else:
                        print(f'Validating: {v} ({s}-{t}, {m}, remaining: {g.shape[0]})' + (f' - ({row["annotator"]}, {row["movement"]}, [{s}, {t}], [{row["start_frame"]}, {row["end_frame"]}])' if self.debug else ''))
                        frames = self.gen_video(video_path, skeleton, s, t)
                        task = self.executor.submit(lambda: self.validate())
                        self.play(v, frames, done=task.done)
                        status, notes = task.result()
                        if row['movement'] == 'NoAction' and status == Status.STEREOTYPICAL:
                            notes = f'{s},{t}'
                        self.df.loc[idx, [self.status_col, self.notes_col, self.timestep_col]] = [status, notes, datetime.now()]
                    self.save()
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
    df = pd.read_csv(r'Z:\Users\TalBarami\videos_qa\qa.csv')
    df = df[(df['start_time'] <= df['end_time']) & (df['start_frame'] <= df['end_frame'])]
    print(f'Total number of samples: {len(df)}')
    # df = df[(df['status'] != "NONE") & (df['status_qa2'] != "NONE")]
    print(f'Number of qa samples: {len(df)}')
    df['status'] = df['status'].apply(lambda s: 'Stereotypical' if s == 'Status.STEREOTYPICAL' else 'NoAction' if s == 'Status.NO_ACTION' else 'NONE')
    df['status_qa2'] = df['status_qa2'].apply(lambda s: 'Stereotypical' if s == 'Status.STEREOTYPICAL' else 'NoAction' if s == 'Status.NO_ACTION' else 'NONE')
    df = df[df['movement'] != 'NoAction']
    df['_annotator'] = df['annotator'].apply(lambda a: 'Human' if a != 'JORDI' else a)
    db = pd.read_csv(DB_PATH)
    props = {v: db[db['basename'] == v][['width', 'height', 'fps', 'frame_count', 'length_seconds']].iloc[0].to_dict() for v, v_path in df[['video', 'video_path']].drop_duplicates().values}
    # df = df[df['child_id'] != 645433144]
    res = pd.DataFrame(columns=['video', 'human_start', 'human_end', 'jordi_start', 'jordi_end',
                               'human_annotation', 'jordi_annotation', 'qa_hadas', 'qa_ofri',
                               'model', 'annotator', 'assessment', 'child_id', 'video_path', 'skeleton_path',
                                'width', 'height', 'fps', 'frame_count', 'length_seconds'])
    _intervals = {v: intervals for v, intervals in df.groupby('video')[['start_frame', 'end_frame']]}
    epsilon = 1e-5
    for i, row in df.iterrows():
        s, t, a, intervals = row['start_frame'] + epsilon, row['end_frame'] - epsilon, row['_annotator'], _intervals[row['video']]
        _res_intervals = {v: intervals for v, intervals in res.groupby('video')[['human_start', 'human_end', 'jordi_start', 'jordi_end']]}
        if row['video'] in _res_intervals:
            res_intervals = _res_intervals[row['video']].fillna(-1)
            idxs = [idx for h_intersection, j_intersection, idx in ((get_intersection((s, t), h_interval), get_intersection((s, t), j_interval), idx)
                                                                    for h_interval, j_interval, idx in zip(res_intervals[['human_start', 'human_end']].values, res_intervals[['jordi_start', 'jordi_end']].values, res_intervals.index))
                    if (h_intersection is not None or j_intersection is not None) and (idx != i)]
            if any(idxs):
                continue

        idxs = [idx for intersection, _, idx in ((get_intersection((s, t), interval), interval, idx) for interval, idx in zip(intervals.values, intervals.index)) if (intersection is not None) and (idx != i)]
        _df = df.loc[idxs]
        same = _df[_df['_annotator'] == a]
        opposite = _df[_df['_annotator'] != a]
        if not same.empty:
            same = same.iloc[0]
            row['start_frame'] = min(row['start_frame'], same['start_frame'])
            row['end_frame'] = max(row['end_frame'], same['end_frame'])
        if opposite.empty:
            r = None
        else:
            r = opposite.iloc[0].copy()
            if opposite.shape[0] > 1:
                m1, m2 = 'Stereotypical' if len(opposite['status'].unique()) > 1 else 'NoAction', 'Stereotypical' if len(opposite['status_qa2'].unique()) > 1 else 'NoAction'
                r[['start_frame', 'end_frame', 'status', 'status_qa2']] = [opposite['start_frame'].min(), opposite['end_frame'].max(), m1, m2]
        if a == 'JORDI':
            m_start, m_end, m_ann = row['start_frame'], row['end_frame'], row['movement']
            if opposite.empty:
                h_start, h_end, h_ann, ann = None, None, 'NoAction', 'Human'
            else:
                h_start, h_end, h_ann, ann = r['start_frame'], r['end_frame'], r['movement'], r['annotator']
        else:
            h_start, h_end, h_ann, ann = row['start_frame'], row['end_frame'], row['movement'], row['annotator']
            if opposite.empty:
                m_start, m_end, m_ann = None, None, 'NoAction'
            else:
                m_start, m_end, m_ann = r['start_frame'], r['end_frame'], r['movement']
        res.loc[res.shape[0]] = ([row['video'], h_start, h_end, m_start, m_end, h_ann, m_ann, row['status'], row['status_qa2'], row['model'], ann,
                                  row['assessment'], row['child_id'], row['video_path'], row['skeleton_path'],
                                  props[row['video']]['width'], props[row['video']]['height'], props[row['video']]['fps'], props[row['video']]['frame_count'], props[row['video']]['length_seconds']])
    res['qa_hadas'] = res.apply(lambda row: 'Stereotypical' if (row['qa_hadas'] == 'NoAction' and row['human_annotation'] != 'NoAction' and row['jordi_annotation'] != 'NoAction') else row['qa_hadas'], axis=1)
    s, no_acts = 0, []
    for v, df in res.groupby('video'):
        df['ss'] = df[['human_start', 'jordi_start']].min(axis=1)
        df['tt'] = df[['human_end', 'jordi_end']].max(axis=1)
        df = df.sort_values(by='ss')
        s = 0
        for i, row in df.iterrows():
            ss, tt = row['ss'], row['tt']
            no_acts.append([row['video'], s, ss, s, ss, 'NoAction', 'NoAction', 'NONE', 'NONE', '', 'Human', row['assessment'], row['child_id'], row['video_path'], row['skeleton_path'], row['width'], row['height'], row['fps'], row['frame_count'], row['length_seconds']])
            s = tt
        no_acts.append([row['video'], s, row['length_seconds'], s, row['length_seconds'], 'NoAction', 'NoAction', 'NONE', 'NONE', '', 'Human', row['assessment'], row['child_id'], row['video_path'], row['skeleton_path'], row['width'], row['height'], row['fps'], row['frame_count'], row['length_seconds']])
    no_acts = pd.DataFrame(no_acts, columns=res.columns)
    res = pd.concat([res, no_acts])
    res['_human_annotation'] = res['human_annotation']
    res['human_annotation'] = res['human_annotation'].apply(lambda s: 'Stereotypical' if s != 'NoAction' else s)
    print(pd.crosstab(res['human_annotation'], res['jordi_annotation']))
    print(pd.crosstab(res['human_annotation'], res['qa_hadas']))
    print(pd.crosstab(res['human_annotation'], res['qa_ofri']))
    print(pd.crosstab(res['jordi_annotation'], res['qa_hadas']))
    print(pd.crosstab(res['jordi_annotation'], res['qa_ofri']))
    # res['movement_bin'] = res['movement'].apply(lambda m: 0 if 'NoAction' in m else 1)
    # res['post_qa'] = res['status'].apply(lambda s: 1 if 'STEREOTYPICAL' in s else 0)
    res.to_csv(r'Z:\Users\TalBarami\videos_qa\qa_processed.csv', index=False)

    # df = df[df['source'] == 'JORDI']
    # df = df[df['movement_bin'] == 1]

def load_dataframe(qa_files):
    root = r'Z:\Users\TalBarami\jordi_cross_validation'
    models = ['cv0.pth', 'cv1.pth', 'cv2.pth', 'cv3.pth', 'cv4.pth']
    dfs = []
    for m in models:
        df = collect_labels(root, osp.join('jordi', m))
        df['model'] = m
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    human = pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\combined.csv')
    human['assessment'] = human['video'].apply(lambda s: '_'.join(s.split('_')[:-2]))
    human['child_id'] = human['assessment'].apply(lambda s: s.split('_')[0]).astype(int)
    human['source'] = 'Human'
    df = pd.concat([df, human], ignore_index=True)
    control = set(pd.read_excel(r'Z:\Users\TalBarami\jordi_cross_validation\control_list.xlsx')['Control children'].tolist())
    df = df[~df['child_id'].isin(control)]
    df['skeleton_path'] = df['video'].apply(lambda v: osp.join(root, v, 'jordi', f'{v}.pkl'))
    df['movement'] = df['movement'].fillna('Stereotypical')
    df = df.groupby(['video', 'start_frame', 'end_frame']).agg({'movement': lambda l: ','.join(set(it.chain(*[[m.strip() for m in x.split(',')] for x in l]))), **{c: lambda x: x.iloc[0] for c in df.columns if c != 'movement'}}).reset_index(drop=True)
    df['status'] = Status.NONE.value
    df['notes'] = ''
    df['qa_timestep'] = None
    df['status_qa2'] = Status.NONE.value
    df['notes_qa2'] = ''
    df['qa2_timestep'] = None
    for qa_file in qa_files:
        if osp.exists(qa_file):
            qa = pd.read_csv(qa_file)
            for i, qa_row in qa.iterrows():
                df_row = df.loc[(df['video'] == qa_row['video']) & (df['start_frame'].round() == np.round(qa_row['start_frame'])) & (df['end_frame'].round() == np.round(qa_row['end_frame']))]
                if len(df_row) == 1:
                    if qa_row['status'] != 'NONE':
                        df.loc[df_row.index, ['status', 'notes', 'qa_timestep']] = qa_row['status'], qa_row['notes'], qa_row['qa_timestep']
                    if qa_row['status_qa2'] != 'NONE':
                        df.loc[df_row.index, ['status_qa2', 'notes_qa2', 'qa2_timestep']] = qa_row['status_qa2'], qa_row['notes_qa2'], qa_row['qa2_timestep']
                elif len(df_row) > 1:
                    print(f'Found {len(df_row)} rows for {qa_row["video"]} {qa_row["start_frame"]}')
                    movements = set(','.join(df_row['movement']).split(','))
                    if len(movements) > 1 and 'Stereotypical' in movements:
                        movements.remove('Stereotypical')
                    movements = ','.join(movements)
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
    df['start_frame'] = df['start_frame'].round()
    df['end_frame'] = df['end_frame'].round()
    df = df.drop_duplicates(subset=['video', 'start_frame']).sort_values(by=['model', 'video', 'start_frame']).reset_index(drop=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-c', '--conclude', action='store_true')
    parser.add_argument('-r', '--reload', action='store_true')
    parser.add_argument('-q', '--qa', action='store_true')
    args = parser.parse_args()
    if args.conclude:
        conclusions()
    else:
        qa_files = [r'Z:\Users\TalBarami\videos_qa\qa.csv', r'Z:\Users\TalBarami\videos_qa\qa_old.csv']
        if args.reload:
            df = load_dataframe(qa_files)
        else:
            df = pd.read_csv(r'Z:\Users\TalBarami\videos_qa\qa.csv')
        ann = Reannotator(df, COCO_LAYOUT, debug=args.debug, qa=args.qa)
        ann.run()
