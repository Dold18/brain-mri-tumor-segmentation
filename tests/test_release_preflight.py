"""Tests for the pre-publication tracked-file safety check."""

from scripts.release_preflight import check_paths


def test_release_preflight_flags_checkpoint_patient_data_and_secrets(tmp_path):
    for path in ('runs/baseline/best.pt', 'data/Task01_BrainTumour/imagesTr/a.nii.gz',
                 '.env', 'cache/a.npy'):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'example')
    findings = check_paths(tmp_path, [
        'runs/baseline/best.pt', 'data/Task01_BrainTumour/imagesTr/a.nii.gz',
        '.env', 'cache/a.npy',
    ])
    assert len(findings) >= 5


def test_release_preflight_allows_code_docs_and_derived_figures(tmp_path):
    paths = ['README.md', 'brain_mri/model.py', 'figures/heldout_BRATS_419.png',
             'results/benchmark_summary.json', 'results/release_audit.json']
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'public')
    assert check_paths(tmp_path, paths) == []


def test_release_preflight_flags_large_files(tmp_path):
    big = tmp_path / 'large.zip'
    with big.open('wb') as f:
        f.truncate(26 * 1024 * 1024)
    assert any('large tracked file' in issue for issue in check_paths(tmp_path, ['large.zip']))
