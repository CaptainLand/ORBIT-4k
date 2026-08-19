import json

from orbit4k.web.app import dataset_status


def test_dataset_state_requires_explicit_completion(tmp_path):
    index = tmp_path / "index.jsonl"
    summary = tmp_path / "summary.json"
    state = tmp_path / "dataset_state.json"

    missing = dataset_status(tmp_path)
    assert not missing["ready"]
    assert missing["status"] == "missing"

    index.write_text("{}\n", encoding="utf-8")
    summary.write_text(json.dumps({"accepted_charts": 1}), encoding="utf-8")
    state.write_text(json.dumps({"status": "building", "processed": 1}), encoding="utf-8")

    building = dataset_status(tmp_path)
    assert not building["ready"]
    assert building["index_exists"]
    assert building["summary_exists"]

    state.write_text(json.dumps({"status": "complete", "processed": 1}), encoding="utf-8")
    complete = dataset_status(tmp_path)
    assert complete["ready"]
    assert complete["status"] == "complete"
