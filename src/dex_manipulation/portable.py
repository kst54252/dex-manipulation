"""Install the explicitly published execution inputs without changing local runs."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest(root):
    path = Path(root) / 'runtime/manifest.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get('schema') != 1:
        raise ValueError('Unsupported runtime bundle schema')
    return data


def relocate_path(path, root):
    """Rebase only explicitly declared source roots; leave saved metadata intact."""
    path, root = Path(path), Path(root).resolve()
    if not path.is_absolute() or path.is_relative_to(root):
        return path
    data = manifest(root)
    for old in (() if data is None else data['source_roots']):
        if path.is_relative_to(old):
            relative = path.relative_to(old)
            if '..' in relative.parts:
                raise ValueError('Invalid archived project path')
            return root / relative
    return path


def _target(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('local',):
        raise ValueError(f'Runtime input must stay under local/: {name}')
    target = (root / name).resolve()
    if not target.is_relative_to(root / 'local'):
        raise ValueError(f'Runtime input escapes local/: {name}')
    return target


def prepare(root, *, verify=False):
    """Restore missing files only. Keep existing user files and archived mtimes.

    Manifest SHA256 checks protect copied bytes, not authenticity of a Git remote.
    Checkpoints and references remain byte-identical to their original contracts.
    """
    root = Path(root).resolve()
    data = manifest(root)
    if data is None:
        return dict(installed=0, files=0, conflicts=[])
    archives = data['archives']
    expected = {}
    for archive, info in archives.items():
        if Path(archive).name != archive:
            raise ValueError('Runtime archive must be a basename')
        for name, item in info['files'].items():
            _target(root, name)
            if name in expected and expected[name] != item:
                raise ValueError(f'Conflicting runtime input: {name}')
            expected[name] = item
    missing = {name for name in expected if not _target(root, name).is_file()}
    installed = 0
    for archive, info in archives.items():
        names = set(info['files'])
        if not verify and not names.intersection(missing):
            continue
        packed = root / 'runtime' / archive
        if not packed.is_file() or sha256(packed) != info['sha256']:
            raise ValueError(f'Missing or damaged runtime archive: {packed}')
        with zipfile.ZipFile(packed) as source:
            if len(source.namelist()) != len(names) or set(source.namelist()) != names:
                raise ValueError(f'Runtime archive inventory mismatch: {archive}')
            for name in sorted(names & missing):
                item = info['files'][name]
                payload = source.read(name)
                if len(payload) != item['size'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
                    raise ValueError(f'Damaged runtime input: {name}')
                target = _target(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive publication avoids replacing concurrent/local work.
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                    temp.write(payload)
                    temporary = Path(temp.name)
                try:
                    os.utime(temporary, ns=(item['mtime_ns'], item['mtime_ns']))
                    try:
                        os.link(temporary, target)
                        installed += 1
                    except FileExistsError:
                        pass
                finally:
                    temporary.unlink()
    conflicts = []
    if verify:
        for name, item in expected.items():
            target = _target(root, name)
            if not target.is_file() or target.stat().st_size != item['size'] or sha256(target) != item['sha256']:
                conflicts.append(name)
    return dict(installed=installed, files=len(expected), conflicts=conflicts)
