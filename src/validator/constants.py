from pathlib import Path
from os import path as osp
from taltools.logging.print_logger import PrintLogger
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESOURCES_ROOT = osp.join(PROJECT_ROOT, 'resources')

logger = PrintLogger(name='Validator', show=False)