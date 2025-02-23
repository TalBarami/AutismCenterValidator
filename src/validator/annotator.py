import atexit
import signal
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from os import path as osp

from validator.constants import logger


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
    def __init__(self, root, annotator_id):
        self.root = root
        self.annotator_id = annotator_id

        self.status_types = self.init_status_types()
        self.data_handler = self.init_data_handler()
        self.video_player = self.init_video_player()

        self.executor = ThreadPoolExecutor()
        self.globals = {g.name: g for g in [Global('exit', 0, self.exit),
                                            Global('revert', 0, self.data_handler.revert),
                                            Global('speed', 1, self.video_player.set_speed),
                                            Global('resolution', 2, self.video_player.set_resolution),
                                            Global('reset', 0, self.video_player.reset)]}
        self.queue_size = 6
        atexit.register(self.data_handler.save)
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    @abstractmethod
    def init_data_handler(self):
        pass

    @abstractmethod
    def init_status_types(self):
        pass

    @abstractmethod
    def init_video_player(self):
        pass


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

    @abstractmethod
    def validate(self, opts, **kwargs):
        pass

    @abstractmethod
    def add_to_queue(self, row):
        pass

    def run(self):
        df = self.data_handler.get_rows()
        n = df.shape[0]
        m = self.data_handler.df.shape[0]

        tasks = [self.executor.submit(self.add_to_queue, df.iloc[i]) for i in range(min(self.queue_size, n))]
        # wait for tasks to finish

        while not df.empty:
            row = df.iloc[0]
            v, s, t = row['basename'], row['start'], row['end']
            idx = row.name
            try:
                logger.info(f'Processing {row["basename"]} {s}-{t}')
                _idx, frames, _args = tasks[0].result()
                if frames is None:
                    df = df.drop(idx)
                    tasks.pop(0)
                    tasks.append(self.executor.submit(self.add_to_queue, df.iloc[min(self.queue_size - 1, n)]))
                    continue
                assert _idx == idx
                task = self.executor.submit(lambda: self.validate(opts=self.status_types, **_args))
                self.video_player.play(f'{v}: ({s}-{t})', frames, done=task.done, counter_text=f'{m-n}/{m}')
                result = task.result()
                self.data_handler.add(idx, result)
                df = df.drop(idx)
                tasks.pop(0)
                n = df.shape[0]
                tasks.append(self.executor.submit(self.add_to_queue, df.iloc[min(self.queue_size-1, n)]))
            except GlobalCommandEvent as gce:
                self.globals[gce.method].func_ref(*gce.args)
