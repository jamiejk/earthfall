"""Preserve the exact declared recipe source bytes before editing or resuming it."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from touchdown.authored_scene import digest
from touchdown.flight_regression import read, write


def capture(run):
    manifest = read(run / 'experiment.json')
    hashes = manifest.get('identity', manifest)['code_hashes']
    folder = run / 'code-snapshot'
    folder.mkdir(exist_ok=True)
    records = []
    for original, expected in hashes.items():
        source = Path(original)
        target = folder / (expected + '-' + source.name)
        if target.exists():
            if digest(target) != expected:
                raise ValueError('Captured recipe code changed')
        else:
            if digest(source) != expected:
                raise ValueError(f'Cannot capture changed recipe source: {source}')
            shutil.copy2(source, target)
            if digest(target) != expected:
                raise ValueError('Recipe copy did not match its recorded hash')
        records.append(dict(original=original, captured=str(target.relative_to(run)), sha256=expected))
    write(folder / 'index.json', dict(files=records, scope='Exact source files declared in experiment code_hashes'))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(f'Preserved {len(capture(args.run.resolve()))} recipe source files')


if __name__ == '__main__':
    main()
