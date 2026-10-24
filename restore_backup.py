"""Verify and restore the complete USB backup from GitHub Release parts."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Only bounded streaming reads are supported')
        chunks = []
        remaining = size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open('rb')
            block = self.current.read(remaining)
            if not block:
                self.current.close()
                self.current = None
                continue
            chunks.append(block)
            remaining -= len(block)
        return b''.join(chunks)

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.parts_dir / 'backup_manifest.json').read_text(encoding='utf-8'))
    out = args.out.resolve()
    target = out / 'autodl_backup'
    if target.exists():
        raise SystemExit(f'Refusing to overwrite existing directory: {target}')
    parts = []
    for entry in manifest['parts']:
        path = args.parts_dir / entry['name']
        if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise SystemExit(f'Part verification failed: {path}')
        print(f'PART VERIFIED {path.name}', flush=True)
        parts.append(path)
    out.mkdir(parents=True, exist_ok=True)
    expected = {x['path']: x for x in manifest['files']}
    seen = set()
    with PartsReader(parts) as stream:
        with tarfile.open(fileobj=stream, mode='r|', bufsize=8 * 1024 * 1024) as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != 'autodl_backup':
                    raise SystemExit(f'Unsafe archive path: {member.name}')
                destination = out.joinpath(*name.parts)
                if not destination.resolve().is_relative_to(target):
                    raise SystemExit(f'Path escapes restore directory: {member.name}')
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise SystemExit(f'Unexpected archive entry: {member.name}')
                entry = expected[member.name]
                if member.size != entry['size']:
                    raise SystemExit(f'Unexpected size: {member.name}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with archive.extractfile(member) as source, destination.open('xb') as dest:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        dest.write(block)
                        h.update(block)
                if h.hexdigest() != entry['sha256']:
                    raise SystemExit(f'File verification failed: {member.name}')
                seen.add(member.name)
                print(f'FILE VERIFIED {member.name}', flush=True)
    if seen != set(expected):
        raise SystemExit(f'Missing files: {sorted(set(expected) - seen)}')
    print(f'Restored and verified {len(seen)} files to {target}')


if __name__ == '__main__':
    main()
