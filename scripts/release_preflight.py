"""Reject accidentally tracked MRI volumes, checkpoints, secrets and large files.

Run from a Git repository root before pushing. This only checks *tracked* files;
it does not read the raw data on the mounted external SSD.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

MAX_SIZE_BYTES = 25 * 1024 * 1024
SENSITIVE_PARTS = frozenset({
    '.venv', '__pycache__', '.env', 'data', 'cache', 'checkpoints', 'runs',
    'patient_data', 'private',
})
SENSITIVE_SUFFIXES = ('.nii', '.nii.gz', '.pt', '.pth', '.h5', '.hdf5', '.dcm', '.dicom')
SENSITIVE_BASENAMES = frozenset({'.env', '.env.local', '.env.production', 'id_rsa', 'id_ed25519'})


def check_paths(root: Path, relative_paths: list[str]) -> list[str]:
    """Return human-readable release blockers for tracked paths."""
    failures: list[str] = []
    for rel in relative_paths:
        rel_path = Path(rel)
        if rel_path.is_absolute() or '..' in rel_path.parts:
            failures.append(f'unsafe tracked path: {rel}')
            continue
        parts = rel_path.parts
        if any(part in SENSITIVE_PARTS for part in parts[:-1]):
            failures.append(f'private/generated directory tracked: {rel}')
        filename = rel_path.name.lower()
        if filename in SENSITIVE_BASENAMES or filename.endswith(SENSITIVE_SUFFIXES):
            failures.append(f'sensitive extension/filename tracked: {rel}')
        path = root / rel_path
        if path.is_file() and path.stat().st_size > MAX_SIZE_BYTES:
            failures.append(f'large tracked file (>25 MiB): {rel}')
    return failures


def tracked_files(root: Path, *, staged: bool = False) -> list[str]:
    cmd = ['git', '-C', str(root), 'ls-files', '-z', '--cached']
    if staged:
        cmd = ['git', '-C', str(root), 'diff', '--cached', '--name-only', '-z', '--diff-filter=ACMR']
    result = subprocess.run(cmd, capture_output=True, check=True)
    return [p.decode('utf-8', 'replace') for p in result.stdout.split(b'\0') if p]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--staged', action='store_true', help='Only inspect added or modified staged paths')
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / '.git').exists():
        raise SystemExit('Not a Git repository; run after git init, from the project root.')
    issues = check_paths(root, tracked_files(root, staged=args.staged))
    if issues:
        print('Release blockers:')
        for issue in issues:
            print(f'  - {issue}')
        raise SystemExit(1)
    print('Tracked-file preflight passed: no sensitive paths or oversized files detected.')


if __name__ == '__main__':
    main()
