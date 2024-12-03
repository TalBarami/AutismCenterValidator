import argparse
from os import path as osp

from taltools.cv.images import COLORS
from taltools.io.files import read_pkl

from validator.annotate_child.yolo_player import YOLOPlayer
from validator.annotate_skeleton.skeleton_annotation_data import SkeletonAnnotationData
from validator.annotator import Annotator
from validator.constants import RESOURCES_ROOT
from validator.data_handler import DataHandler
from validator.video_player import VideoPlayer


class SkeletonChildAnnotator(Annotator):

    def init_status_types(self):
        return ['OK', 'Child detected as adult', 'Adult detected as child', 'Missing skeleton of the child', 'Mixup', 'Skip']

    def init_data_handler(self):
        return SkeletonAnnotationData(annotator_id=self.annotator_id,
                                      test_dir=osp.join(self.root, 'testing'))

    def init_video_player(self):
        return VideoPlayer(osp.join(RESOURCES_ROOT, 'config.json'))

    def validate(self, opts):
        ans = self.choose('Choose status:', opts)
        status = opts[ans[0]]
        result_notes = input('Save notes: ') if status == 'Skip' else None
        return status, result_notes

    def add_to_queue(self, row):
        processed = self.video_player.gen_video(row['video_path'])
        return row.name, processed, {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotator')
    parser.add_argument('--root', type=str, help='Root directory')
    parser.add_argument('--annotator', type=int, help='Annotator ID', choices=[0, 1])
    args = parser.parse_args()
    annotators = {0: 'noa', 1: 'shaked'}
    root = args.root
    annotator = args.annotator
    print('Starting annotator...')
    program = SkeletonChildAnnotator(root, annotator)
    print('Annotator started. Short guide:')
    print(f'1. Choose status: {", ".join(program.status_types)}')
    print('2. If you typed \'Skip\', you will be asked to save notes.')
    print('3. To exit annotator, type \'-exit\'.')
    print('4. To revert last action, type \'-revert\'.')
    print('5. To change video speed, type \'-speed <speed>\'.')
    print('6. To change video resolution, type \'-resolution <width> <height>\'.')
    print('7. To restart the video, type \'-reset\'.')
    program.run()
