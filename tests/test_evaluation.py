import importlib.util
import json
from pathlib import Path

import pytest

from safety.evaluation import (
    Sample,
    break_even_checks,
    engine_report,
    expected_gate,
    gate_metrics,
    item_metrics,
    load_labels,
    usd_per_check,
)
from safety.inspector import DENY, RETRY, UNLOCK
from safety.jev_client import PpeAnswer

ITEMS = ("hard_hat", "vest", "boots")


def sample(label, choice, conf=0.9, item="hard_hat", image="a.jpg", latency=0.1):
    return Sample(image, item, label, PpeAnswer(item, choice, conf, latency), True)


def test_load_labels_maps_hat_column_and_skips_empty(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("image, Hat ,vest,boots\nw1.jpg,yes,NO,\nw2.jpg,unclear,yes,yes\n,yes,yes,yes\n",
                    encoding="utf-8")
    assert load_labels(path) == [
        ("w1.jpg", {"hard_hat": "yes", "vest": "no"}),
        ("w2.jpg", {"hard_hat": "unclear", "vest": "yes", "boots": "yes"}),
    ]


def test_load_labels_rejects_unknown_values(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("image,hat\nw1.jpg,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_labels(path)


def test_metrics_count_false_positives_and_unclear():
    samples = [
        sample("yes", "yes"),            # correct
        sample("yes", "unclear", 0.0),   # unclear
        sample("no", "yes"),             # false positive: missing but said present
        sample("no", "no"),              # correct
        sample("unclear", "yes"),        # a person couldn't tell: not scored
    ]
    m = item_metrics(samples, 0.7)
    assert m["labelled"] == 4
    assert m["accuracy"] == 0.5
    assert m["accuracy_when_answered"] == round(2 / 3, 4)
    assert m["false_positive_rate"] == 0.5
    assert m["false_negative_rate"] == 0.0
    assert m["unclear_rate"] == 0.25
    assert m["confusion"]["no"] == {"yes": 1, "no": 1, "unclear": 0}
    assert m["human_unclear"]["yes"] == 1


def test_low_confidence_yes_is_scored_as_unclear_like_the_gate():
    m = item_metrics([sample("no", "yes", conf=0.6)], 0.7)
    assert m["false_positive_rate"] == 0.0 and m["unclear_rate"] == 1.0


def test_gate_metrics_flag_false_unlocks():
    assert expected_gate({"hard_hat": "yes", "vest": "no", "boots": "yes"}, ITEMS) == DENY
    assert expected_gate({"hard_hat": "yes", "vest": "yes", "boots": "yes"}, ITEMS) == UNLOCK
    assert expected_gate({"hard_hat": "yes", "vest": "unclear", "boots": "yes"}, ITEMS) is None
    g = gate_metrics([(DENY, UNLOCK), (DENY, DENY), (UNLOCK, UNLOCK), (UNLOCK, RETRY), (None, UNLOCK)])
    assert g["people"] == 4 and g["people_missing_ppe"] == 2
    assert g["false_unlocks"] == 1 and g["false_unlock_rate"] == 0.5
    assert g["retry_rate"] == 0.25 and g["accuracy"] == 0.5


def test_engine_report_latency_ignores_crops_never_sent():
    samples = [sample("yes", "yes", item=i, latency=0.2) for i in ITEMS]
    samples.append(Sample("b.jpg", "boots", "yes", PpeAnswer("boots", "unclear", 0.0), called=False))
    r = engine_report(samples, [(UNLOCK, UNLOCK)], 0.7, ITEMS)
    assert r["latency"]["per_call"]["count"] == 3
    assert r["latency"]["per_person_sequential"]["p50_ms"] == pytest.approx(600, abs=0.5)
    assert r["per_item"]["vest"]["accuracy"] == 1.0
    empty = engine_report([], [], 0.7, ITEMS)
    assert empty["overall"]["accuracy"] is None and empty["latency"]["per_call"] == {"count": 0}
    assert empty["tokens"] == {"calls_counted": 0}


def jev_sample(item, image, tokens):
    return Sample(image, item, "yes", PpeAnswer(item, "yes", 0.9, 0.3, input_tokens=tokens), True)


def test_tokens_are_summed_per_worker_check():
    samples = [jev_sample(i, "a.jpg", 100) for i in ITEMS] + [jev_sample(i, "b.jpg", 200) for i in ITEMS]
    tokens = engine_report(samples, [], 0.7, ITEMS)["tokens"]
    assert tokens == {"calls_counted": 6, "mean_input_per_call": 150.0, "mean_input_per_worker_check": 450.0}


def test_cost_and_break_even():
    assert usd_per_check(500, 2.0) == pytest.approx(0.001)
    assert break_even_checks(1.0, 0.001) == pytest.approx(1000)
    assert break_even_checks(0.0, 0.001) == 0      # local adds no cost: cheaper at any volume
    assert break_even_checks(1.0, 0.0) is None     # Jev free: local never cheaper


def test_post_figures_writes_table_and_cost_graph(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "post_figures", Path(__file__).resolve().parent.parent / "tools" / "post_figures.py")
    post = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post)
    jev = [jev_sample(i, "a.jpg", 400) for i in ITEMS]
    local = [sample("yes", "unclear", 0.0, item=i) for i in ITEMS]
    report = json.loads(json.dumps({
        "items": list(ITEMS),
        "dataset": {"evaluated": 1},
        "engines": {"jev_vlm": engine_report(jev, [(UNLOCK, UNLOCK)], 0.7, ITEMS),
                    "local_ml": engine_report(local, [(UNLOCK, RETRY)], 0.5, ITEMS)},
    }))
    labels, columns = post.table_rows(report, 2.0)
    assert columns["jev_vlm"][labels.index("Input tokens per worker check")] == "1200"
    assert columns["jev_vlm"][-1] == "$2.40" and columns["local_ml"][-1] == "$0 extra"
    post.draw_table(report, 2.0, tmp_path / "table.png")
    assert post.draw_cost(0.0024, 0.6, tmp_path / "cost.png") == pytest.approx(250)
    assert (tmp_path / "table.png").stat().st_size and (tmp_path / "cost.png").stat().st_size
