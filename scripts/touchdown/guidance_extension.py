"""Original 97 guide frames, one untimed appearance reference, no image pins."""
from pathlib import Path
import argparse
import json
import os
import shutil

from touchdown.authored_scene import digest
from touchdown.flight_regression import read, write, verify_snapshot
from touchdown.local_descent import graph as local_graph, validate_graph as validate_local

SOURCE=Path('outputs/touchdown/studio/studio-chillon-minimal-clay-v2-20261026').resolve()
PHOTO=Path('outputs/touchdown/studio/studio-chillon-neutral-ladder-v1-20261027/view-2.png').resolve()
ROOT=Path('outputs/touchdown/studio/studio-chillon-guidance-extension-v1-20261027').resolve()
PROMPT=('One uninterrupted photoreal aerial camera flight. Starting with a vast regional '
        'view of a broad blue-green Swiss lake and surrounding land, the camera flies '
        'forward and descends continuously along the shore, then settles at eye level '
        'toward a stationary adult woman beside a gravel path. She only becomes '
        'resolvable as the camera gets near. She wears a beige knee-length trench coat '
        'and black shoes, with brown hair and natural adult anatomy. Her feet remain '
        'planted on the same grass; her arms rest at her sides during approach and '
        'open outward during the final close settle. Lake left, meadow and route right, '
        'a distant historic pale stone castle behind-left. Weathered masonry, irregular '
        'autumn foliage, photographic grass, rippling water and fabric in consistent '
        'natural daylight. Foreground and distant scenery change perspective through '
        'the continuous physical camera move.')


def graph(guide,photo):
    g=local_graph(guide,photo)
    g['308']['inputs']['length']=97
    g['305']['inputs']['text']=PROMPT
    for node in ['90','377']:
        if node in g:g[node]['inputs']['filename_prefix']=g[node]['inputs']['filename_prefix'].replace('local_descent_v1','guidance_extension_v1')
    validate(g)
    return g


def validate(g):
    if g['308']['inputs']['length']!=97:raise ValueError('Original 97-frame guide clock required')
    # Uploaded SDK file references contain locks and cannot be deep-copied.
    # Only the latent-length input needs normalization for the unit validator.
    unit=dict(g)
    unit['308']={**g['308'],'inputs':{**g['308']['inputs'],'length':49}}
    validate_local(unit)
    if g['305']['inputs']['text']!=PROMPT:raise ValueError('Frozen phase-scoped prompt changed')
    if g['340']['inputs']['image']!=['60',0]:raise ValueError('Stage 2 lost motion guide')


def board(root,video,name):
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    c=cv2.VideoCapture(str(video));total=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    sheet=Image.new('RGB',(1440,888),(24,24,24));d=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/google-noto/NotoSans-Regular.ttf',16)
    folder=root/(name+'-samples');folder.mkdir(exist_ok=True)
    for n,i in enumerate([round(k*(total-1)/8) for k in range(9)]):
        c.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=c.read()
        if not ok:raise ValueError('Missing frame')
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.save(folder/f'{i:04d}.png')
        x=n%3*480;y=n//3*296;sheet.paste(im.resize((480,272)),(x,y+24))
        d.text((x+8,y+2),f'{name} {i}/96 | delivery {(96+i)/24:.2f}s',font=font,fill='white')
    c.release();sheet.save(root/(name+'-sheet.jpg'),quality=94)


def prepare(root):
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh root required')
    root.mkdir(parents=True,exist_ok=True)
    old=read(SOURCE/'raw-guide-review.json')
    assert old['status']=='pass' and old['guide_sha256']==digest(SOURCE/'guide.mp4')
    assert old['geometry_sha256']==digest(SOURCE/'geometry-audit.json')
    verify_snapshot(old['stage_layout_hashes'])
    for name in ['guide.mp4','render.json','geometry-audit.json','raw-guide-review.json']:
        shutil.copy2(SOURCE/name,root/name)
    shutil.copy2(PHOTO,root/'appearance.png')
    files=[SOURCE/n for n in ['guide.mp4','render.json','geometry-audit.json','raw-guide-review.json']]
    files += [PHOTO,Path(__file__).resolve(),Path('scripts/touchdown/local_descent.py').resolve(),Path('src/geoblock_core/workflow.py').resolve()]
    write(root/'experiment.json',dict(frames=97,fps=24,seed=1399629552,prompt=PROMPT,
        source_hashes={str(p):digest(p) for p in files},delivery_mapping=list(range(96,193)),
        original_clock=dict(bounds=[0,29,58,108,156,192],fps=24,unchanged=True),
        appearance_reference='Neutral ground view 2 as IC index -1 in both stages; not a timed endpoint',
        changed_from_local=['Existing full regional/approach/settle grey guide instead of newly registered local guide',
                            'Ground view 2 instead of elevated view 3 for castle/subject appearance',
                            'Prompt describes the original phases and arm opening'],
        retained=['No timed image pins','Full layout and appearance in both stages','Same weights/seed/8+3 recipe'],
        limitations=['Old guide and photograph only approximately registered','Broad castle block may impose its shape',
                     'Not an isolated duration-only comparison','No orbit or cloud guidance in this source'],
        scope='Original delivery-frame 96–192 segment only; no full-film acceptance or promotion'))
    snapshot=root/'code-snapshot';snapshot.mkdir()
    for p in files:
        if p.suffix=='.py':shutil.copy2(p,snapshot/p.name)
    board(root,root/'guide.mp4','guide')
    write(root/'status.json',dict(status='prepared_awaiting_guide_review',app_promoted=False))


def run(root):
    for line in Path('/tmp/touchdown-generation-runtime.env').read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('TOUCHDOWN_'):os.environ[key]=json.loads(value)
    from touchdown import studio_build as studio
    from touchdown.layout_recipe_trial import setup_runtime,metadata
    from geoblock_core.workflow import LTX_VIDEO,STAGE1_VIDEO
    import fakeout_earth_zoom as f
    if (root/'submitted-graph.json').exists():raise ValueError('Attempt frozen; no reroll')
    verify_snapshot(read(root/'experiment.json')['source_hashes'])
    review=read(root/'guide-review.json');assert review['status']=='pass'
    verify_snapshot(review['hashes'])
    runtime=root/'runtime';runtime.mkdir(exist_ok=False);setup_runtime(SOURCE,runtime)
    custom=Path('outputs/review/neutral-ladder-20261027/runtime_nodes').resolve()
    config=runtime/'pinned-model-paths.json';paths=read(config)
    paths['mask_attention_compatibility']={'custom_nodes':str(custom)};write(config,paths)
    shutil.copy2(custom/'TouchdownMaskAttention/__init__.py',runtime/'mask-attention-shim.py')
    with studio.runtime('comfy',runtime), f._client(studio.VIDEO_URL) as client:
        a=client.assets.from_file(str(root/'guide.mp4'));b=client.assets.from_file(str(root/'appearance.png'))
        g=graph(a,b);write(root/'submitted-graph.json',json.loads(json.dumps(g,default=str)))
        write(root/'status.json',dict(status='running',app_promoted=False))
        saved=f._submit(client,g,(LTX_VIDEO,STAGE1_VIDEO),root/'ltx')
        for name,node,size in [('result',LTX_VIDEO,(1920,1088)),('stage1-result',STAGE1_VIDEO,(960,544))]:
            p=Path(saved[node]);meta=metadata(p)
            assert (meta['width'],meta['height'])==size
            write(root/(name+'.json'),dict(video=str(p),sha256=digest(p),**meta));board(root,p,name)
    shutil.copy2(read(root/'result.json')['video'],root/'result.mp4')
    write(root/'status.json',dict(status='held_awaiting_visual_review',app_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    (prepare if a.stage=='prepare' else run)(a.root.resolve())
