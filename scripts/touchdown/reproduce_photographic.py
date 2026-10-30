"""One command to prepare and run the frozen Chambord/Peña flight recipe."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

from touchdown.photographic_recipe import prepare,environment,REPO
from touchdown.flight_runtime import capture_runtime
from touchdown.flight_regression import read,write
from touchdown.authored_scene import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Fresh output directory')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    prepare(root)
    subprocess.run([sys.executable,str(REPO/'scripts/touchdown/photographic_recipe_assets.py'),'assets',str(root)],check=True)
    environment()
    runtime=root/'runtime-record';runtime.mkdir()
    capture_runtime(runtime)
    models=read(REPO/'outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026/model-fingerprints.json')
    for record in models.values():
        stat=Path(record['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(record['size'],record['mtime_ns']):
            raise ValueError('Installed pinned weights changed')
    write(runtime/'model-fingerprints.json',models)
    write(root/'launcher.json',dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__)),
        asset_helper_sha256=digest(REPO/'scripts/touchdown/photographic_recipe_assets.py'),
        prepare_only=a.prepare_only))
    if a.prepare_only:
        print('Prepared and checked; no inference jobs submitted')
        return
    env=dict(os.environ);env['PYTHONPATH']=''
    with (root/'worker.log').open('w') as log:
        subprocess.run([sys.executable,str(REPO/'scripts/touchdown/photographic_recipe.py'),'worker','--root',str(root)],
                       env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(root/'run/motion/earth_zoom.mp4')


if __name__=='__main__':main()
