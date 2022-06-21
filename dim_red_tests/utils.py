import random
import time
from os.path import join as pjoin
from pathlib import Path

import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
RESULTS_FOLDER = pjoin(PROJECT_ROOT, "results")


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time() - ts
        print(f"func '{func.__name__}' took {te} sec")
        return result

    return timed


def seed_everything(seed=314159) -> None:
    random.seed(seed)
    np.random.seed(seed)
