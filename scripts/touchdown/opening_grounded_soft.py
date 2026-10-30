"""One controlled correction: grounded opening's intermediate image strength 1 -> 0.5."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

spec=importlib.util.spec_from_file_location('grounded_opening_driver',Path(__file__).with_name('opening_grounded.py'))
grounded=importlib.util.module_from_spec(spec);spec.loader.exec_module(grounded)
base=grounded.base


def prepare(root):
    grounded.prepare(root)
    plan=base.read(root/'experiment.json')
    plan['extra_photographic_guide']['strength']=.5
    plan['comparison']='Grounded opening first pass held/jumped; intermediate guide strength alone reduced from 1.0 to 0.5'
    plan['controlled_correction']=dict(parameter='Intermediate generation-frame-40 guide strength',before=1.0,after=.5,
        retained='Same grounded images, crop, prompts, seed, frames, models, cloud guide and later footage')
    plan['hashes'][str(Path(__file__).resolve())]=base.sha(Path(__file__))
    base.write(root/'experiment.json',plan)


def worker(root):
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    code=root/'frozen-code';sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    from touchdown import split_flight as flight
    generate=flight._generate

    def softer(f,client,graph,frames,pins,strengths,timing,guides,seed,prefix,folder,options):
        assert frames==49 and strengths=={40:1.0}
        return generate(f,client,graph,frames,pins,{40:.5},timing,guides,seed,prefix,folder,options)

    flight._generate=softer
    grounded.pinned.worker(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','worker','assemble'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.stage=='assemble':base.assemble(root)
    else:globals()[a.stage](root)
