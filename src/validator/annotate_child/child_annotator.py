import argparse
from os import path as osp

from taltools.cv.images import COLORS
from taltools.io.files import read_pkl

from validator.annotate_child.child_annotation_data import ChildAnnotationData
from validator.annotate_child.yolo_player import YOLOPlayer
from validator.annotator import Annotator
from validator.constants import RESOURCES_ROOT


class ChildAnnotator(Annotator):
    def __init__(self, root, annotator_id, annotations_filename):
        self.annotations_filename = annotations_filename
        super().__init__(root, annotator_id)

    def init_status_types(self):
        return ['OK', 'Child present but not detected', 'No Child', 'Overlapping', 'Corrupted video', 'Skip']

    def init_data_handler(self):
        return ChildAnnotationData(annotations_file=osp.join(self.root, self.annotations_filename),
                                   annotations_dir=osp.join(self.root, 'annotations'),
                                   data_dir=osp.join(self.root, 'data'),
                                   annotator_id=self.annotator_id)

    def init_video_player(self):
        return YOLOPlayer(osp.join(RESOURCES_ROOT, 'config.json'))

    def validate(self, opts, max_people):
        ans = self.choose('Choose status:', opts)
        status = opts[ans[0]]
        child_ids = self.choose('Choose child ID(s):', [COLORS[i % len(COLORS)]['name'] for i in range(max_people)], offset=0) if status == 'OK' else None
        result_notes = input('Save notes: ') if status == 'Skip' else None
        return {'status': status, 'notes': result_notes, 'child_ids': child_ids}

    def add_to_queue(self, row):
        if not osp.exists(row['data_path']):
            return row.name, None, None
        tracking = read_pkl(row['data_path'])
        a = [x['boxes'].id.max() for x in tracking['data'] if x['boxes'].id is not None]
        k = int(max(a) if len(a) > 0 else 0) + 1
        processed = self.video_player.gen_video(row['video_path'], tracking)
        return row.name, processed, {'max_people': k}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotator')
    parser.add_argument('--root', type=str, help='Root directory')
    parser.add_argument('--annotator', type=int, help='Annotator ID', choices=[0, 1])
    args = parser.parse_args()
    annotators = {0: 'noa', 1: 'shaked'}
    root = args.root
    annotator = args.annotator
    print('Starting annotator...')
    program = ChildAnnotator(root, annotator, f'{annotators[annotator]}.csv')
    print('Annotator started. Short guide:')
    print(f'1. Choose status: {", ".join(program.status_types)}')
    print('2. If you typed \'Skip\', you will be asked to save notes.')
    print('3. If you typed \'OK\', you will be asked to choose child ID(s). You can choose multiple IDs separated by space.')
    print('4. To exit annotator, type \'-exit\'.')
    print('5. To revert last action, type \'-revert\'.')
    print('6. To change video speed, type \'-speed <speed>\'.')
    print('7. To change video resolution, type \'-resolution <width> <height>\'.')
    print('8. To restart the video, type \'-reset\'.')
    program.run()
