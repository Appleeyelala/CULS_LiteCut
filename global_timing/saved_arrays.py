"""Stream arrays in our pretty-printed saved JSON, including gzip archives.

The input contract is the repository's indent=2 top-level object serialization;
this is not a general arbitrary-JSON query engine. Never restores an archive in
place or modifies historical artifacts.
"""
import gzip
import json
from pathlib import Path
import re


def iter_saved_array(path, field, chunk_size=65536):
    path=Path(path)
    if not path.exists() and path.with_suffix(path.suffix+'.gz').exists():
        path=path.with_suffix(path.suffix+'.gz')
    opener=gzip.open if path.suffix=='.gz' else open
    marker=re.compile(r'(?:^|\n)  '+re.escape(json.dumps(field))+r'\s*:\s*\[')
    decoder=json.JSONDecoder()
    with opener(path,'rt',encoding='utf-8') as stream:
        buf=''; eof=False
        while True:
            block=stream.read(chunk_size)
            if not block: raise ValueError(f'Missing top-level array {field} in {path}')
            buf+=block
            match=marker.search(buf)
            if match:
                buf=buf[match.end():];break
            buf=buf[-max(128,len(field)*2):]
        first=True
        while True:
            buf=buf.lstrip()
            if not buf and not eof:
                block=stream.read(chunk_size);eof=not block;buf+=block;continue
            if buf.startswith(']'):return
            if not first:
                if not buf.startswith(','):raise ValueError('Missing array separator or truncated JSON.')
                buf=buf[1:].lstrip()
            while True:
                buf=buf.lstrip()
                try:
                    obj,end=decoder.raw_decode(buf)
                    # A complete JSON scalar could straddle a chunk. This API
                    # only accepts object entries, matching catalogue/scores.
                    if not isinstance(obj,dict):raise ValueError('Expected object array entry.')
                    break
                except json.JSONDecodeError as exc:
                    block=stream.read(chunk_size)
                    if not block:raise ValueError('Truncated or malformed saved array.') from exc
                    buf+=block
                    if len(buf)>64*1024**2:raise ValueError('Unexpectedly large saved object.')
            yield obj
            buf=buf[end:];first=False
