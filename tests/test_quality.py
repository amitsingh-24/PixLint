from __future__ import annotations


from pixlint.analysis.quality import analyze_quality


class TestQualityAnalysis:
    def test_quality_blur(self, folder_dataset):
        report = analyze_quality(folder_dataset, metrics=["blur"])
        assert report.dataset_id == folder_dataset.dataset_id
        assert len(report.per_image) == len(folder_dataset)
        for qi in report.per_image:
            assert qi.blur_score is not None
            assert qi.blur_label in ("sharp", "acceptable", "blurry", "good", "poor")

    def test_quality_exposure(self, folder_dataset):
        report = analyze_quality(folder_dataset, metrics=["exposure"])
        assert len(report.per_image) == len(folder_dataset)
        for qi in report.per_image:
            assert qi.exposure_score is not None
            assert qi.exposure_label in ("good", "underexposed", "overexposed", "acceptable", "poor")

    def test_quality_noise(self, folder_dataset):
        report = analyze_quality(folder_dataset, metrics=["noise"])
        for qi in report.per_image:
            assert qi.noise_score is not None

    def test_quality_contrast(self, folder_dataset):
        report = analyze_quality(folder_dataset, metrics=["contrast"])
        for qi in report.per_image:
            assert qi.contrast_score is not None

    def test_quality_resolution(self, folder_dataset):
        report = analyze_quality(folder_dataset, metrics=["resolution"])
        for qi in report.per_image:
            assert qi.resolution_score is not None

    def test_quality_color(self, folder_dataset):
        report = analyze_quality(folder_dataset, metrics=["color"])
        for qi in report.per_image:
            assert qi.color_score is not None

    def test_quality_all_metrics(self, folder_dataset):
        report = analyze_quality(folder_dataset, metrics=["blur", "exposure", "noise", "contrast", "resolution", "color"])
        assert report.average_overall is not None
        assert report.flagged_images >= 0

    def test_quality_report_structure(self, folder_dataset):
        report = analyze_quality(folder_dataset)
        assert hasattr(report, "per_image")
        assert hasattr(report, "average_overall")
        assert hasattr(report, "flagged_images")
        assert hasattr(report, "summary")
