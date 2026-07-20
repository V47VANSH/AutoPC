"""Task ledger and store: persistence, plan discipline, criteria, attempts."""
import json
import os

import pytest

from autopc.tasks import TaskLedger, TaskStore


@pytest.fixture
def store(tmp_path):
    return TaskStore(str(tmp_path / "tasks"))


def test_create_persists_goal_verbatim(store):
    t = store.create("Open Notepad and type: hello, world!")
    reloaded = TaskLedger.load(t.path)
    assert reloaded.goal == "Open Notepad and type: hello, world!"
    assert reloaded.status == "active"
    assert os.path.exists(t.journal_file)


def test_empty_goal_rejected(store):
    with pytest.raises(ValueError):
        store.create("   ")


def test_plan_allows_only_one_step_in_progress(store):
    t = store.create("x")
    t.set_plan([{"text": "a", "status": "in_progress"},
                {"text": "b", "status": "in_progress"},
                {"text": "c", "status": "bogus"},
                {"text": "", "status": "pending"}])
    assert [p.status for p in t.plan] == ["in_progress", "pending", "pending"]
    assert t.current_step() == "a"


def test_current_step_falls_back_to_first_pending(store):
    t = store.create("x")
    t.set_plan([{"text": "a", "status": "done"}, {"text": "b", "status": "pending"}])
    assert t.current_step() == "b"


def test_criteria_marking_and_unmet(store):
    t = store.create("x")
    t.set_criteria(["window open", "text typed"])
    assert [c.id for c in t.criteria] == ["c1", "c2"]
    assert not t.all_marked_met()
    assert t.mark_criterion("c1", True, "title bar says Notepad")
    assert [c.id for c in t.unmet()] == ["c2"]
    assert not t.mark_criterion("c9", True)
    t.mark_criterion("c2", True, "hello visible")
    assert t.all_marked_met()


def test_new_attempt_resets_stale_met_marks(store):
    t = store.create("x")
    t.set_criteria(["a"])
    t.begin_attempt()
    t.mark_criterion("c1", True, "looked done")
    t.end_attempt("exhausted")
    t.begin_attempt()
    assert t.criteria[0].status == "pending"
    assert t.current_attempt.n == 2


def test_user_blocker_is_scoped_to_current_attempt(store):
    t = store.create("x")
    t.begin_attempt()
    t.add_blocker("needs a password", needs_user=True)
    assert t.has_user_blocker()
    t.end_attempt("needs_user")
    t.begin_attempt()
    assert not t.has_user_blocker()


def test_round_trip_preserves_everything(store):
    t = store.create("x")
    t.set_criteria(["a"])
    t.set_plan([{"text": "s1", "status": "in_progress"}])
    t.begin_attempt()
    t.log("did a thing")
    t.lessons.append("use the keyboard shortcut")
    t.save()
    r = TaskLedger.load(t.path)
    assert r.to_dict() == t.to_dict()


def test_save_is_atomic_no_temp_files_left(store):
    t = store.create("x")
    for i in range(5):
        t.log(f"step {i}")
    leftovers = [n for n in os.listdir(t.path) if n.endswith(".tmp")]
    assert leftovers == []
    with open(t.ledger_file, encoding="utf-8") as f:
        json.load(f)  # valid JSON


def test_anchor_names_goal_unmet_criteria_and_step(store):
    t = store.create("Book the 9am meeting room")
    t.set_criteria(["room shows as booked"])
    t.set_plan([{"text": "open calendar", "status": "in_progress"}])
    a = t.anchor()
    assert "Book the 9am meeting room" in a
    assert "room shows as booked" in a
    assert "open calendar" in a


def test_store_prefix_lookup_and_ordering(store):
    a = store.create("first task")
    b = store.create("second task", status="queued")
    c = store.create("third task", status="queued")
    assert store.load(a.id).goal == "first task"
    assert store.load(a.id[:-3]) is None or store.load(a.id[:-3]).id == a.id
    assert store.next_queued().id == b.id          # FIFO
    assert store.latest_resumable().id == a.id     # only `a` was started
    c.finish("verified")
    c.save()
    assert {t.id for t in store.unfinished()} == {a.id, b.id}


def test_duplicate_ids_get_suffixed(store):
    a = store.create("same goal")
    b = store.create("same goal")
    assert a.id != b.id
