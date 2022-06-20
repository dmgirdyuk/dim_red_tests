import time
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time() - ts
        print(f"{te} sec")
        return result

    return timed
