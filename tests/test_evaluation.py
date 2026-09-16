from concurrent.futures import ThreadPoolExecutor
import threading

from evaluation.runner import TruthsCache, _run_metric


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


def test_truths_cache_miss_returns_none():
    cache = TruthsCache()
    assert cache.get("missing") is None


def test_truths_cache_get_or_compute_first_write_wins():
    cache = TruthsCache()
    first, is_owner_first = cache.get_or_compute("t1", lambda: ["truth-a"])
    second, is_owner_second = cache.get_or_compute("t1", lambda: ["truth-b"])

    assert first == ["truth-a"]
    assert second == ["truth-a"]
    assert is_owner_first is True
    assert is_owner_second is False


def test_truths_cache_computes_exactly_once_under_concurrency():
    cache = TruthsCache()
    calls = {"count": 0}
    lock = threading.Lock()

    def producer():
        with lock:
            calls["count"] += 1
        return ["shared truth"]

    def worker():
        return cache.get_or_compute("t1", producer)[0]

    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(lambda _: worker(), range(3)))

    assert calls["count"] == 1
    assert values == [["shared truth"]] * 3


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
    cache.get_or_compute("t1", lambda: ["cached truth"])
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
