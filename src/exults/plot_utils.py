"""
Usage example:
hehe = 0
@live_update()
def train_loss():
    global hehe
    hehe += 1
    return lp.ggplot({'x': hehe + np.arange(10), 'y': np.arange(10)}) + lp.geom_line(lp.aes(x='x', y='y'))

train_loss.stop()
"""

import time
import threading
import IPython.display as idis
import json


all_handlers = set()


def stop_all():
    while all_handlers:
        all_handlers.pop().stop()


class LiveHandler:
    def __init__(self):
        self.okay = True

    def stop(self):
        self.okay = False
        self.thread.join()


def live_update(period=10):
    # see https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.DisplayHandle
    def wrapper(plotter):
        t = idis.display(plotter(), display_id=True)
        handler = LiveHandler()

        def loop():
            while handler.okay:
                time.sleep(period)
                t.update(plotter())

        thread = threading.Thread(target=loop)
        thread.start()
        handler.thread = thread
        all_handlers.add(handler)
        return handler

    return wrapper


def read_jsonl(log_file):
    with open(log_file) as f:
        train_accs = f.read().strip().split("\n")
        train_accs = [json.loads(a) for a in train_accs]
    return train_accs


def annotate(l_d, **kwargs):
    return [{**d, **kwargs} for d in l_d]


def label_floats(l_f, label):
    return [{label: f} for f in l_f]


def label_it(l_d, label):
    return [{**d, label: it} for it, d in enumerate(l_d)]


def compose(*funcs):
    def composed(l_d):
        for f in funcs:
            l_d = f(l_d)
        return l_d

    return composed
