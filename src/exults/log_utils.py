import os
import json
import re
from pathlib import Path

def slugify(value):    
    value = re.sub(r"[^\w\s-]", "_", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_cache_path(cache_path, *args):
    return os.path.join(cache_path, *[slugify(arg) for arg in args])


def cached(name, thing, prefixes, force_compute=False):
    import torch
    file_path = get_cache_path(*prefixes, name) + ".pt"
    if not force_compute and os.path.exists(file_path):
        print(f"using cached {file_path}")
        return torch.load(file_path)
    else:
        if not os.path.exists(os.path.dirname(file_path)):
            print(f"making cache path {os.path.dirname(file_path)}")
            os.makedirs(os.path.dirname(file_path))
        result = thing()
        torch.save(result, file_path)
        return result

class LogSeq:
    def __init__(self, path, mode):
        assert mode in ['append']
        self.path = path
        self.mode = mode
        self.data = []
    def append(self, x):
        self.data.append(x)
        if self.mode == 'append':
            with open(self.path, 'a') as f:
                f.write('\n')
                f.write(json.dumps(x))
    def reset(self):
        with open(self.path, 'w') as f:
            f.write('')
        self.data = []
    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)
    
class Logger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_handles = {}
        os.makedirs(log_path, exist_ok=True)
        
    def init(self, name, mode='append'):
        assert name not in self.log_handles
        self.log_handles[name] = LogSeq(
            path=self.log_path / (slugify(name) + '.jsonl'),
            mode=mode
        )
    def __getattr__(self, x):
        if x == "log_handles":
            # handles infinite recursion during pickling
            raise AttributeError("This shouldn't happen")
        if x in self.log_handles:
            return self.log_handles[x]
        raise AttributeError(f'Unknown log handle {x}')
        
        
    def reset(self):
        for _, handle in self.log_handles.items():
            handle.reset()
        self.log_handles = {}
        