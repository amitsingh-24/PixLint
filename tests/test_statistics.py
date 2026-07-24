from __future__ import annotations

from pixlint.analysis.statistics import compute_statistics, sample_dataset


class TestSampling:
    def test_random_sample(self, folder_dataset):
        sampled = sample_dataset(folder_dataset, n=3, strategy="random")
        assert len(sampled) == 3
        assert all(isinstance(s, str) for s in sampled)

    def test_sample_more_than_available(self, folder_dataset):
        sampled = sample_dataset(folder_dataset, n=100, strategy="random")
        assert len(sampled) == len(folder_dataset)

    def test_stratified_sample(self, coco_dataset):
        sampled = sample_dataset(coco_dataset, n=4, strategy="stratified")
        assert len(sampled) <= 4

    def test_deterministic_seed(self, folder_dataset):
        s1 = sample_dataset(folder_dataset, n=3, seed=42)
        s2 = sample_dataset(folder_dataset, n=3, seed=42)
        assert s1 == s2


class TestStatistics:
    def test_statistics_structure(self, folder_dataset):
        stats = compute_statistics(folder_dataset)
        assert stats.num_images == 5
        assert stats.num_annotations == 0
        assert stats.num_classes == 0

    def test_statistics_with_annotations(self, coco_dataset):
        stats = compute_statistics(coco_dataset)
        assert stats.num_images == 4
        assert stats.num_annotations == 4
        assert stats.num_classes == 2

    def test_statistics_image_sizes(self, folder_dataset):
        stats = compute_statistics(folder_dataset)
        assert stats.min_image_size[0] > 0
        assert stats.max_image_size[0] > 0
        assert stats.avg_image_size[0] > 0
