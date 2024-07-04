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
    def __init__(self, root, annotations_filename='annotations.csv'):
        self.root = root
        self.status_types = ['OK', 'No child', 'Overlapping', 'Corrupted video', 'Skip']

        self.data_handler = DataHandler(osp.join(root, annotations_filename), osp.join(root, 'data2'))
        self.video_player = YOLOPlayer(osp.join(RESOURCES_ROOT, 'config.json'))

        self.executor = ThreadPoolExecutor()
        self.globals = {g.name: g for g in [Global('exit', 0, self.exit),
                                            Global('revert', 0, self.data_handler.revert),
                                            Global('speed', 1, self.video_player.set_speed),
                                            Global('resolution', 2, self.video_player.set_resolution),
                                            Global('reset', 0, self.video_player.reset)]}

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

    def run(self):
        df = self.data_handler.get_rows()
        n = df.shape[0]
        m = self.data_handler.df.shape[0]
        while not df.empty:
            row = df.iloc[0]
            v, s, t = row['video'], row['start_frame'], row['end_frame']
            idx = row.name
            try:
                logger.info(f'Processing {row["video"]} {s}-{t}')
                tracking = read_pkl(row['data_path'])
                # k = int(skeleton['person_ids'].max() + 1)
                k = max([int(x['boxes'].id.max()) for x in tracking['data'] if x['boxes'].id is not None])
                processed = self.video_player.gen_video(row['video_path'], tracking)
                task = self.executor.submit(lambda: self.validate(opts=self.status_types, max_people=k))
                self.video_player.play(f'{v}: ({s}-{t})', processed, done=task.done, counter_text=f'{m-n}/{m}')
                status, notes, child_ids = task.result()
                self.data_handler.add(idx, status, notes, child_ids)
                df = df.drop(idx)
                n = df.shape[0]
            except GlobalCommandEvent as gce:
                self.globals[gce.method].func_ref(*gce.args)


if __name__ == '__main__':
    ann = Annotator(r'Z:\Users\TalBarami\ChildDetect', 'noa.csv')
    ann.run()
