"""Transfer the frozen photographic recipe to the single pre-recorded random case."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys
REPO=Path(__file__).resolve().parents[2]
SELECTION=REPO/'outputs/review/photographic-random-transfer-20261027/selection.json'
ROOT=REPO/'outputs/touchdown/studio/studio-eilean-donan-photographic-transfer-v1-20261027'


def prepare(root):
    from touchdown.photographic_recipe import prepare as base_prepare,environment
    from touchdown.flight_regression import read,write
    from touchdown.authored_scene import digest
    from touchdown.flight_runtime import capture_runtime
    selected=read(SELECTION);source=Path(selected['selected']['source'])
    base_prepare(root)
    shutil=__import__('shutil');shutil.copy2(SELECTION,root/'selection.json')
    recipe=read(root/'recipe.json')
    baseline=read(REPO/'outputs/touchdown/studio/studio-chambord-photographic-recipe-v1-20261027/recipe.json')
    if recipe['full_split_settings']!=baseline['full_split_settings']:raise ValueError('Frozen recipe changed')
    cfg=read(source/'scene-config.json')
    if cfg.get('video',{}).get('phase_contract'):raise ValueError('This transfer uses historical instructions unchanged')
    recipe.update(source=str(source),case=selected['selected'],selection_sha256=digest(SELECTION),
        scope='One random new-landmark flight-stage transfer with existing photographs/portrait; no seed search or per-place tuning')
    recipe['inputs_and_code'].update({str(Path(__file__).resolve()):digest(Path(__file__)),str((root/'selection.json').resolve()):digest(root/'selection.json')})
    write(root/'recipe.json',recipe)
    subprocess.run([sys.executable,str(REPO/'scripts/touchdown/photographic_recipe_assets.py'),'assets',str(root)],check=True)
    environment();runtime=root/'runtime-record';runtime.mkdir();capture_runtime(runtime)
    models=read(REPO/'outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026/model-fingerprints.json')
    for record in models.values():
        s=Path(record['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(record['size'],record['mtime_ns']):raise ValueError('Pinned model changed')
    write(runtime/'model-fingerprints.json',models)
    write(root/'transfer-policy.json',dict(recipe_settings_identical=True,inference_code_unchanged=True,
        source=str(source),seed=cfg['seed'],parent_seed=cfg['seed']+32,descent_seed=cfg['seed']+33,
        changes=['Destination photographic inputs, native portrait and recorded scene seed'],
        retained=['Archived inference code','Full photographic-flight-v1 preset','Existing approximately 2km aerial instruction','Historical assembly/hold trimming'],
        added_30000ft_constraint=False,extra_image_pins=False,creative_rerolls=0,app_promoted=False))
    print('Prepared transfer:',cfg['place'],flush=True)


def worker(root):
    # Do not import the live touchdown package in this process. The unchanged
    # worker installs the archived package before it imports inference code.
    source=Path(json.loads((root/'recipe.json').read_text())['source'])
    spec=importlib.util.spec_from_file_location('frozen_recipe_driver',REPO/'scripts/touchdown/photographic_recipe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.SOURCE=source
    module.worker(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker']);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args();(prepare if a.stage=='prepare' else worker)(a.root.resolve())
