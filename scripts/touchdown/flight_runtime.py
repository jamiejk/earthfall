"""Capture the actual local runtime and recipe, without modifying model services."""
from pathlib import Path
import json
import os
import shlex
import subprocess
from touchdown.authored_scene import digest

REPO = Path(__file__).resolve().parents[2]


def recipe_hashes():
    files = set((REPO/'scripts/touchdown').glob('*.py'))
    files.update((REPO/'src/geoblock_core').rglob('*.py'))
    files.update((REPO/'scripts/touchdown/presets').glob('*.json'))
    files.update(REPO/'scripts'/p for p in ['fakeout_earth_zoom.py', 'start_touchdown_vision_checker.sh'])
    return {str(p): digest(p) for p in sorted(files)}


def runtime_identity():
    cwd = Path(os.environ['TOUCHDOWN_COMFY_SERVER_CWD']).resolve()
    command = shlex.split(os.environ['TOUCHDOWN_COMFY_SERVER_CMD'])
    versions = subprocess.check_output([command[0], '-c',
        'import json,sys,importlib.metadata as m; print(json.dumps(dict(python=sys.version,packages={d.metadata["Name"]:d.version for d in m.distributions() if d.metadata["Name"]})))'], text=True)
    files = set((cwd/'comfy').rglob('*.py')) | set((cwd/'comfy_extras').rglob('*.py'))
    files.update(cwd/p for p in ['main.py','nodes.py','node_helpers.py','folder_paths.py'])
    pinned = REPO/'outputs/touchdown/relay-runtime/custom_nodes/TouchdownPromptRelay'
    if not pinned.is_dir():  # a fresh clone has only the copy vendored with the canonical workflow
        pinned = REPO/'workflows/touchdown/canonical/photographic-grounded-v1/relay-runtime/TouchdownPromptRelay'
    files.update(pinned/p for p in ['patches.py','prompt_relay.py','upstream.json'])
    if any(not p.is_file() for p in files):
        raise ValueError('Runtime source is incomplete')
    return dict(command=command,cwd=str(cwd),versions=json.loads(versions),
                source_hashes={str(p):digest(p) for p in sorted(files)},
                adapter_sha256=digest(REPO/'scripts/touchdown/relay_node.py'))


def capture_runtime(root):
    identity=dict(recipe=recipe_hashes(),runtime=runtime_identity())
    target=root/'runtime-identity.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError('Recipe or runtime changed: use a fresh run')
    target.write_text(json.dumps(identity,indent=2)+'\n')
    return identity
