"""Tests for the H5 embedding dataset loader and fold generation."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from enso_purity_mil.dataset import EmbeddingBagDataset, custom_collate_fn
from enso_purity_mil.folds import generate_stratified_folds


# ── Helpers ───────────────────────────────────────────────────────
def _make_h5(path: Path, n_tiles: int = 500, dim: int = 2560):
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=np.random.randn(n_tiles, dim).astype(np.float32))
        coords = np.column_stack([
            np.random.randint(0, 10000, n_tiles),
            np.random.randint(0, 10000, n_tiles),
        ]).astype(np.int32)
        f.create_dataset("coords", data=coords)
        f.attrs["tile_size"] = 224
        f.attrs["target_mpp"] = 0.5


@pytest.fixture()
def sample_manifest(tmp_path) -> tuple[pd.DataFrame, Path]:
    """Build a tiny manifest + H5 files for testing.

    Layout:
      - Aliquot ALQ-0: 2 slides (uuid-0000 100 tiles, uuid-0001 300 tiles)
      - Aliquot ALQ-1: 1 slide  (uuid-0002 500 tiles)
      - Aliquot ALQ-2: 1 slide  (uuid-0003 700 tiles)
      - Normal:        1 slide  (uuid-0004 200 tiles, purity=0.0)
    """
    h5_dir = tmp_path / "embeddings"
    h5_dir.mkdir()
    records = []
    tile_counts = [100, 300, 500, 700, 200]
    aliquots = ["ALQ-0", "ALQ-0", "ALQ-1", "ALQ-2", None]
    purities = [0.8, 0.8, 0.6, 0.4, 0.0]
    match_types = ["same_portion", "same_portion", "same_portion", "same_portion", "normal_tissue"]

    for i in range(5):
        fid = f"uuid-{i:04d}"
        _make_h5(h5_dir / f"{fid}.h5", n_tiles=tile_counts[i])
        records.append({
            "file_uuid_original": fid,
            "barcode": f"TCGA-AA-{i:04d}-01A-0{i+1}-TS1",
            "aliquot_barcode": aliquots[i],
            "purity": purities[i],
            "case_id": f"TCGA-AA-{i:04d}",
            "sample_type_code": "01" if i < 4 else "11",
            "gdc_match_type": match_types[i],
            "cancer_type": "LUAD",
        })
    return pd.DataFrame(records), h5_dir


# ── Dataset ──────────────────────────────────────────────────────
class TestEmbeddingBagDataset:
    def test_length_groups_by_aliquot(self, sample_manifest):
        manifest, h5_dir = sample_manifest
        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=256)
        # 3 aliquots + 1 normal = 4 bags
        assert len(ds) == 4

    def test_multi_slide_aliquot_concatenates(self, sample_manifest):
        manifest, h5_dir = sample_manifest
        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=256)
        # ALQ-0 has 100+300=400 tiles → sample 256 without replacement
        feats, label = ds[0]
        assert feats.shape == (256, 2560)
        assert label == pytest.approx(0.8)

    def test_replacement_sampling_when_pool_small(self, sample_manifest):
        manifest, h5_dir = sample_manifest
        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=4096)
        # ALQ-0 pool = 400 tiles < 4096 → must sample with replacement
        feats, label = ds[0]
        assert feats.shape == (4096, 2560)

    def test_normal_slide_is_own_bag(self, sample_manifest):
        manifest, h5_dir = sample_manifest
        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=64)
        # Last bag is the normal slide
        feats, label = ds[len(ds) - 1]
        assert label == pytest.approx(0.0)
        assert feats.shape == (64, 2560)

    def test_collate(self, sample_manifest):
        manifest, h5_dir = sample_manifest
        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=128)
        batch = [ds[i] for i in range(3)]
        feats, labels = custom_collate_fn(batch)
        assert feats.shape == (3, 128, 2560)
        assert labels.shape == (3,)

    def test_purity_values_bounded(self, sample_manifest):
        manifest, h5_dir = sample_manifest
        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=64)
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0.0 <= label <= 1.0

    def test_smart_io_large_pool_partial_read(self, sample_manifest):
        """When total tiles > num_instances, only sampled rows are read."""
        manifest, h5_dir = sample_manifest
        # ALQ-1 has 500 tiles; request 64 → proportional partial read
        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=64)
        feats, label = ds[1]  # ALQ-1
        assert feats.shape == (64, 2560)
        assert label == pytest.approx(0.6)

    def test_non_sequential_index(self, sample_manifest):
        """Manifest with non-sequential index must not crash."""
        manifest, h5_dir = sample_manifest
        filtered = manifest.iloc[[0, 2, 3, 4]].copy()
        assert list(filtered.index) == [0, 2, 3, 4]
        ds = EmbeddingBagDataset(filtered, h5_dir, num_instances=64)
        for i in range(len(ds)):
            feats, label = ds[i]
            assert feats.shape[0] == 64

    def test_cache_dir_fast_path(self, sample_manifest, tmp_path):
        """Cached .pt files (named by bag_id) are loaded instead of H5."""
        manifest, h5_dir = sample_manifest
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        ds_orig = EmbeddingBagDataset(manifest, h5_dir, num_instances=64)
        for idx in range(len(ds_orig)):
            feats, label = ds_orig[idx]
            bag_id = ds_orig.groups[idx]["bag_id"]
            torch.save({"feats": feats.half(), "label": label}, cache_dir / f"{bag_id}.pt")

        ds_cached = EmbeddingBagDataset(manifest, h5_dir, num_instances=64, cache_dir=cache_dir)
        assert len(ds_cached) == len(ds_orig)
        feats_c, label_c = ds_cached[0]
        assert feats_c.shape == (64, 2560)
        assert feats_c.dtype == torch.float32

    def test_cache_pool_resamples_each_access(self, sample_manifest, tmp_path):
        """New cache format (feats_pool) should produce fresh samples each access."""
        manifest, h5_dir = sample_manifest
        cache_dir = tmp_path / "cache_pool"
        cache_dir.mkdir()

        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=64)
        bag_id = ds.groups[0]["bag_id"]

        # Build a deterministic pool to ensure re-sampling changes row selection.
        base = torch.arange(500 * 2560, dtype=torch.float32).reshape(500, 2560).half()
        torch.save({"feats_pool": base, "label": 0.8}, cache_dir / f"{bag_id}.pt")

        ds_cached = EmbeddingBagDataset(manifest, h5_dir, num_instances=64, cache_dir=cache_dir)
        feats_a, _ = ds_cached[0]
        feats_b, _ = ds_cached[0]
        assert feats_a.shape == (64, 2560)
        assert feats_b.shape == (64, 2560)
        assert not torch.equal(feats_a, feats_b), "Expected different sampled rows across accesses"

    def test_legacy_fixed_cache_falls_back_to_h5_resample(self, sample_manifest, tmp_path):
        """Legacy fixed-size cache should not lock training to one tile subset forever."""
        manifest, h5_dir = sample_manifest
        cache_dir = tmp_path / "cache_legacy"
        cache_dir.mkdir()

        ds = EmbeddingBagDataset(manifest, h5_dir, num_instances=64)
        bag_id = ds.groups[0]["bag_id"]

        # Sentinel tensor should never appear if loader correctly re-samples from H5.
        sentinel = torch.full((64, 2560), 1234.0, dtype=torch.float16)
        torch.save({"feats": sentinel, "label": 0.8}, cache_dir / f"{bag_id}.pt")

        ds_cached = EmbeddingBagDataset(manifest, h5_dir, num_instances=64, cache_dir=cache_dir)
        feats, _ = ds_cached[0]
        assert feats.shape == (64, 2560)
        assert feats.dtype == torch.float32
        assert float(feats.max()) < 100.0, "Expected H5 re-sample, not sentinel cached tensor"

    def test_bag_id_stable_across_folds(self, sample_manifest):
        """bag_id must not change when manifest rows are shuffled."""
        manifest, h5_dir = sample_manifest
        ds1 = EmbeddingBagDataset(manifest, h5_dir, num_instances=64)
        ids1 = {g["bag_id"] for g in ds1.groups}

        shuffled = manifest.sample(frac=1, random_state=99).reset_index(drop=True)
        ds2 = EmbeddingBagDataset(shuffled, h5_dir, num_instances=64)
        ids2 = {g["bag_id"] for g in ds2.groups}

        assert ids1 == ids2, "bag_ids must be deterministic regardless of row order"


# ── Fold generation ──────────────────────────────────────────────
class TestFoldGeneration:
    def test_five_folds(self, sample_manifest):
        manifest, _ = sample_manifest
        folds = generate_stratified_folds(manifest, n_folds=5, seed=42,
                                           cancer_col="cancer_type")
        assert len(folds) == 5
        all_indices = set()
        for fold_indices in folds:
            all_indices.update(fold_indices)
        assert all_indices == set(range(len(manifest)))

    def test_no_patient_leak(self, sample_manifest):
        manifest, _ = sample_manifest
        folds = generate_stratified_folds(manifest, n_folds=5, seed=42,
                                           cancer_col="cancer_type")
        for i in range(5):
            for j in range(i + 1, 5):
                cases_i = set(manifest.iloc[folds[i]]["case_id"])
                cases_j = set(manifest.iloc[folds[j]]["case_id"])
                assert cases_i.isdisjoint(cases_j), f"Patient leak between fold {i} and {j}"

    def test_deterministic(self, sample_manifest):
        manifest, _ = sample_manifest
        f1 = generate_stratified_folds(manifest, n_folds=5, seed=42,
                                        cancer_col="cancer_type")
        f2 = generate_stratified_folds(manifest, n_folds=5, seed=42,
                                        cancer_col="cancer_type")
        for a, b in zip(f1, f2):
            assert a == b

    def test_non_sequential_index(self, sample_manifest):
        """Folds must work on a DataFrame with non-sequential index."""
        manifest, _ = sample_manifest
        gapped = manifest.iloc[[0, 2, 3, 4]].copy()  # index = [0, 2, 3, 4]
        gapped = gapped.reset_index(drop=True)
        folds = generate_stratified_folds(gapped, n_folds=4, seed=42,
                                           cancer_col="cancer_type")
        all_indices = set()
        for fold_indices in folds:
            all_indices.update(fold_indices)
        assert all_indices == set(range(len(gapped)))
