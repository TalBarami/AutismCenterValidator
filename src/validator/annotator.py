import argparse
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from os import path as osp

import pandas as pd
from skeleton_tools.utils.constants import COLORS
from skeleton_tools.utils.tools import read_json, write_json, read_pkl

from validator.constants import RESOURCES_ROOT, logger
from validator.data_handler import DataHandler
from validator.video_player import VideoPlayer, YOLOPlayer


class Global:
    def __init__(self, name, args_count, func_ref):
        self.name = name
        self.args_count = args_count
        self.func_ref = func_ref

class GlobalCommandEvent(Exception):
    def __init__(self, method, *args):
        self.method = method
        self.args = args

    def __str__(self):
        return f'{self.method} {self.args}'


class Annotator:
    def __init__(self, root, annotations_filename, annotator_id):
        self.root = root
        self.annotations_filename = annotations_filename
        self.annotator_id = annotator_id

        self.status_types = ['OK', 'No child', 'Overlapping', 'Corrupted video', 'Skip']
        self.data_handler = DataHandler(osp.join(root, self.annotations_filename), osp.join(root, 'annotations'), osp.join(root, 'data'), self.annotator_id)
        self.video_player = YOLOPlayer(osp.join(RESOURCES_ROOT, 'config.json'))

        self.executor = ThreadPoolExecutor()
        self.globals = {g.name: g for g in [Global('exit', 0, self.exit),
                                            Global('revert', 0, self.data_handler.revert),
                                            Global('speed', 1, self.video_player.set_speed),
                                            Global('resolution', 2, self.video_player.set_resolution),
                                            Global('reset', 0, self.video_player.reset)]}
        self.queue_size = 10

    def exit(self):
        self.data_handler.save()
        self.video_player.update_cfg()
        exit(0)

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

    def validate(self, opts, max_people):
        ans = self.choose('Choose status:', opts)
        status = opts[ans[0]]
        child_ids = self.choose('Choose child ID(s):', [COLORS[i % len(COLORS)]['name'] for i in range(max_people)], offset=0) if status == 'OK' else None
        result_notes = input('Save notes: ') if status == 'Skip' else None
        return status, result_notes, child_ids

    def add_to_queue(self, row):
        if not osp.exists(row['data_path']):
            return row.name, None, None
        tracking = read_pkl(row['data_path'])
        a = [x['boxes'].id.max() for x in tracking['data'] if x['boxes'].id is not None]
        k = int(max(a) if len(a) > 0 else 0) + 1
        processed = self.video_player.gen_video(row['video_path'], tracking)
        return row.name, processed, k

    def run(self):
        df = self.data_handler.get_rows()
        n = df.shape[0]
        m = self.data_handler.df.shape[0]

        tasks = [self.executor.submit(self.add_to_queue, df.iloc[i]) for i in range(min(self.queue_size, n))]
        # wait for tasks to finish

        while not df.empty:
            row = df.iloc[0]
            v, s, t = row['basename'], row['start_frame'], row['end_frame']
            idx = row.name
            try:
                logger.info(f'Processing {row["basename"]} {s}-{t}')
                _idx, frames, k = tasks[0].result()
                if not osp.exists(row['video_path']) or not osp.exists(row['data_path']):
                    df = df.drop(idx)
                    tasks.pop(0)
                    tasks.append(self.executor.submit(self.add_to_queue, df.iloc[min(self.queue_size - 1, n)]))
                    continue
                assert _idx == idx
                task = self.executor.submit(lambda: self.validate(opts=self.status_types, max_people=k))
                self.video_player.play(f'{v}: ({s}-{t})', frames, done=task.done, counter_text=f'{m-n}/{m}')
                status, notes, child_ids = task.result()
                self.data_handler.add(idx, status, notes, child_ids)
                df = df.drop(idx)
                tasks.pop(0)
                n = df.shape[0]
                tasks.append(self.executor.submit(self.add_to_queue, df.iloc[min(self.queue_size-1, n)]))
            except GlobalCommandEvent as gce:
                self.globals[gce.method].func_ref(*gce.args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotator')
    parser.add_argument('--root', type=str, help='Root directory')
    parser.add_argument('--annotator', type=int, help='Annotator ID', choices=[0, 1])
    args = parser.parse_args()
    annotators = {0: 'noa', 1: 'shaked'}
    root = args.root
    annotator = args.annotator
    print('Starting annotator...')
    program = Annotator(root, f'{annotators[annotator]}.csv', annotator)
    print('Annotator started. Short guide:')
    print('1. Choose status: OK, No child, Overlapping, Corrupted video, Skip')
    print('2. If you typed \'Skip\', you will be asked to save notes.')
    print('3. If you typed \'OK\', you will be asked to choose child ID(s). You can choose multiple IDs separated by space.')
    print('4. To exit annotator, type \'-exit\'.')
    print('5. To revert last action, type \'-revert\'.')
    print('6. To change video speed, type \'-speed <speed>\'.')
    print('7. To change video resolution, type \'-resolution <width> <height>\'.')
    print('8. To restart the video, type \'-reset\'.')
    program.run()
