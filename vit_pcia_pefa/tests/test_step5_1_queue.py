from pathlib import Path
import sqlite3

from vit_pcia_pefa.src.step5_1_queue import SEED_NAMESPACE, create_queue, task_seed


def test_step5_1_queue_uses_exact_seed_namespace(tmp_path: Path):
    queue = tmp_path / "queue.sqlite3"
    tasks = [{"task_id": "vit_b16__08", "payload": {"model_id": "vit_b16"}}]
    receipt = create_queue(queue, tasks)
    with sqlite3.connect(queue) as connection:
        seed = connection.execute("SELECT seed FROM tasks").fetchone()[0]
        namespace = connection.execute("SELECT value FROM metadata WHERE key='seed_namespace'").fetchone()[0]
    assert receipt["seed_namespace"] == namespace == SEED_NAMESPACE
    assert int(seed) == task_seed("vit_b16__08")


def test_step5_1_queue_normalizes_flat_manifest_rows(tmp_path: Path):
    queue = tmp_path / "queue.sqlite3"
    create_queue(queue, [{"task_id": "deit_s16__08", "model_id": "deit_s16", "split_order": 8}])
    with sqlite3.connect(queue) as connection:
        payload = __import__("json").loads(connection.execute("SELECT payload_json FROM tasks").fetchone()[0])
    assert payload == {"model_id": "deit_s16", "split_order": 8}
