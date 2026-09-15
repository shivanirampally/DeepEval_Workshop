from deepeval_framework.evaluation import TruthsCache, _run_metric


class FakeMetric:
    def __init__(self, truths_result=None, score=0.9, reason="ok"):
        self.score = score
        self.reason = reason
        self.truths = None
        self.measure_calls = 0
        self._truths_result = truths_result

    def _generate_truths(self, *args, **kwargs):
        if self._truths_result is None:
            raise AssertionError(
                "real truths generation should not run on a cache hit"
            )
        return self._truths_result

    def measure(self, test_case):
        self.measure_calls += 1
        self.truths = self._generate_truths()


def test_truths_cache_set_is_first_write_wins():
    cache = TruthsCache()
    cache.set("t1", ["truth-a"])
    cache.set("t1", ["truth-b"])

    assert cache.get("t1") == ["truth-a"]


def test_truths_cache_miss_returns_none():
    cache = TruthsCache()
    assert cache.get("missing") is None


def test_run_metric_faithfulness_cache_miss_populates_cache():
    cache = TruthsCache()
    metric = FakeMetric(truths_result=["source truth"])

    result = _run_metric(
        "faithfulness", metric, test_case=None,
        truths_cache=cache, cache_key="t1",
    )

    assert result["status"] == "COMPLETED"
    assert metric.measure_calls == 1
    assert cache.get("t1") == ["source truth"]


def test_run_metric_faithfulness_cache_hit_skips_generation():
    cache = TruthsCache()
    cache.set("t1", ["cached truth"])
    metric = FakeMetric(truths_result=None)  # would raise if regenerated

    result = _run_metric(
        "faithfulness", metric, test_case=None,
        truths_cache=cache, cache_key="t1",
    )

    assert result["status"] == "COMPLETED"
    assert metric.truths == ["cached truth"]


def test_run_metric_non_faithfulness_ignores_cache():
    cache = TruthsCache()
    metric = FakeMetric(truths_result=["irrelevant"])

    result = _run_metric(
        "hallucination", metric, test_case=None,
        truths_cache=cache, cache_key="t1",
    )

    assert result["status"] == "COMPLETED"
    assert cache.get("t1") is None
