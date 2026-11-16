"""Resolve lossless JSON references from the extracted ZIP; standard library only."""
import json
from pathlib import Path

def resolve_file(bundle_root, relative_path):
    root = Path(bundle_root).resolve()
    cache = {}
    active = set()
    def load(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Reference escapes bundle')
        if name not in cache:
            cache[name] = json.loads(path.read_text())
        return cache[name]
    def resolve(value):
        if isinstance(value, dict) and set(value) == {'$bundle_ref'}:
            target = value['$bundle_ref']
            key = (target['file'], target['pointer'])
            if key in active:
                raise ValueError('Reference cycle')
            active.add(key)
            obj = load(target['file'])
            for token in target['pointer'].split('/')[1:]:
                obj = obj[int(token)] if isinstance(obj, list) else obj[token.replace('~1', '/').replace('~0', '~')]
            answer = resolve(obj)
            active.remove(key)
            return answer
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(load(relative_path))
