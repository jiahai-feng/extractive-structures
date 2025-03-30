import os.path as osp
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(verbose=True)
ROOT = Path(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
