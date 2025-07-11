import contextlib
from timeit import default_timer
from typing import Callable

import joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report
    into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def measure(func: Callable):

    def inner(*args, **kwargs):
        start = default_timer()
        func(*args, **kwargs)
        elapsed_sec = default_timer - start
        print(f"Elapsed: {elapsed_sec:.3f} secs")

    return inner


def repeat(n: int = 1):
    def decorator(func: Callable):
        def inner(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)

        return inner

    return decorator
