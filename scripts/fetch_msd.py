"""Download and safely extract official MSD Task01_BrainTumour dataset.

Run: python -m scripts.fetch_msd  # uses /Volumes/Storage
Or:  python -m scripts.fetch_msd --archive /path/to/Task01_BrainTumour.tar

A public, non-diagnostic research dataset licensed CC BY-SA 4.0. Download
source and checksum are published by the MONAI project.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

from brain_mri.data import read_catalog
from brain_mri.storage_paths import DATA_ROOT, require_ssd_if_targeted

DATASET_NAME = "Task01_BrainTumour"
ARCHIVE_NAME = f"{DATASET_NAME}.tar"
SOURCE_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
# Published in MONAI DecathlonDataset resource list.
EXPECTED_MD5 = "240a19d752f0d9e9101544901065d872"
BUFFER = 4 * 1024 * 1024


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - compare with provider's published dataset checksum
    with path.open("rb") as handle:
        while block := handle.read(BUFFER):
            digest.update(block)
    return digest.hexdigest()


def download_archive(url: str, destination: Path) -> Path:
    """Resume large HTTP downloads into .part files; reject servers ignoring Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Existing archive: {destination}", flush=True)
        return destination
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "brain-mri-research/0.2"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing:
            raise RuntimeError(
                f"Server rejected resume at byte {existing}; remove {partial} to restart"
            ) from exc
        raise
    with response:
        status = response.status
        if existing and status != 206:
            raise RuntimeError(
                f"Server ignored HTTP Range request (status {status}). "
                f"Preserved {partial}; remove it to restart."
            )
        if existing and not response.headers.get("Content-Range", "").startswith(f"bytes {existing}-"):
            raise RuntimeError("Resume response has an incorrect Content-Range")
        if not existing and status != 200:
            raise RuntimeError(f"Unexpected HTTP status {status} downloading dataset")
        total = response.headers.get("Content-Length")
        remaining = int(total) if total is not None else None
        if remaining is not None:
            free_bytes = shutil.disk_usage(destination.parent).free
            if free_bytes < remaining + 512 * 1024 * 1024:
                raise OSError(
                    f"Not enough free disk for download: need about "
                    f"{(remaining + 512 * 1024 * 1024) / 2**30:.1f} GiB; "
                    f"available {free_bytes / 2**30:.1f} GiB."
                )
        downloaded = 0
        last_print = 0.0
        with partial.open("ab" if existing else "wb") as output:
            while True:
                chunk = response.read(BUFFER)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_print >= 5:
                    print(f"Downloaded {(existing + downloaded) / 2**30:.2f} GiB", flush=True)
                    last_print = now
        if remaining is not None and downloaded != remaining:
            raise OSError(
                f"Incomplete download: expected {remaining} more bytes; got {downloaded}. "
                f"Rerun to resume {partial}."
            )
    os.replace(partial, destination)
    return destination


def _archive_member_path(member: tarfile.TarInfo) -> Path | None:
    # Treat all names as POSIX regardless of the host platform. Ignore optional
    # macOS Finder artifacts; reject malicious paths or archive features.
    raw = member.name
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ValueError(f"Unsafe archive member name: {raw!r}")
    if path.parts and (path.name.startswith("._") or path.name == ".DS_Store"):
        return None
    if not path.parts or path.parts[0] != DATASET_NAME:
        raise ValueError(f"Unexpected directory in MSD archive: {raw!r}")
    if not member.isfile() and not member.isdir():
        raise ValueError(f"Refusing link or special archive entry: {raw!r}")
    return Path(*path.parts)


def extract_archive(archive: Path, output_root: Path) -> Path:
    """Extract to a temporary directory then rename only after integrity checks."""
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / DATASET_NAME
    if target.exists():
        read_catalog(target, require_files=True)
        print(f"Dataset already present and catalog files exist: {target}", flush=True)
        return target
    needed_free = int(archive.stat().st_size * 1.02)
    available = shutil.disk_usage(output_root).free
    if available < needed_free:
        raise OSError(
            f"Insufficient free space to extract archive; "
            f"need about {needed_free / 2**30:.1f} GiB, "
            f"have {available / 2**30:.1f} GiB."
        )
    with tempfile.TemporaryDirectory(prefix=".msd-extract-", dir=output_root) as staging:
        stage = Path(staging)
        seen: set[Path] = set()
        files = 0
        with tarfile.open(archive, mode="r:") as bundle:
            for member in bundle:
                path = _archive_member_path(member)
                if path is None:
                    continue
                if path in seen:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                seen.add(path)
                dest = stage / path
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"Missing file content for {member.name}")
                with source, dest.open("xb") as output:
                    copied = shutil.copyfileobj(source, output, BUFFER)
                    del copied
                if dest.stat().st_size != member.size:
                    raise ValueError(f"Truncated archive member: {member.name}")
                files += 1
                if files % 100 == 0:
                    print(f"Extracted {files} files", flush=True)
        candidate = stage / DATASET_NAME
        read_catalog(candidate, require_files=True)
        if target.exists():
            raise FileExistsError(f"Dataset appeared during extraction: {target}")
        candidate.rename(target)
    print(f"Extracted {files} files to {target}", flush=True)
    return target


def fetch_dataset(output_root: Path, *, archive: Path | None = None,
                  verify_checksum: bool = True) -> Path:
    require_ssd_if_targeted(output_root)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / DATASET_NAME
    if target.is_dir():
        read_catalog(target, require_files=True)
        print(f"Using existing dataset at {target}", flush=True)
        return target
    downloaded = archive is None
    if archive is None:
        archive = download_archive(SOURCE_URL, output_root / "downloads" / ARCHIVE_NAME)
    else:
        archive = archive.expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Archive does not exist: {archive}")
    if verify_checksum:
        print("Verifying official MONAI MD5 checksum...", flush=True)
        actual = md5sum(archive)
        if actual != EXPECTED_MD5:
            suggestion = ("Delete the damaged downloaded archive and rerun." if downloaded else
                          "Check that this is the official MSD Task01 archive.")
            raise ValueError(f"Dataset archive checksum mismatch ({actual}). {suggestion}")
    result = extract_archive(archive, output_root)
    print("Dataset ready. License: CC BY-SA 4.0; cite Medical Segmentation Decathlon.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT,
                        help="Destination containing Task01_BrainTumour (may be external SSD)")
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded local TAR")
    args = parser.parse_args()
    print(fetch_dataset(args.output_root, archive=args.archive))


if __name__ == "__main__":
    main()
