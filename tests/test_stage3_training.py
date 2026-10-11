import pytest

from echolens.stage3_training import epoch_batches, second_epoch_decision


def test_epoch_has_every_record_once_and_keeps_partial_batch():
    batches = epoch_batches(810, 0)
    assert len(batches) == 51 and len(batches[-1]) == 10
    assert sorted(i for batch in batches for i in batch) == list(range(810))
    assert batches == epoch_batches(810, 0)
    assert batches != epoch_batches(810, 1)
    # A checkpoint after step25 resumes at batch25, without replaying committed data.
    assert [i for b in batches[:25] + epoch_batches(810, 0)[25:] for i in b] == [i for b in batches for i in b]


@pytest.mark.parametrize('initial,mid,end,expected', [
    (0.3, 0.2, 0.19, True), (0.3, 0.2, 0.199, False),
    (0.3, 0.2, 0.21, False), (0.1, 0.2, 0.19, False)])
def test_second_epoch_policy(initial, mid, end, expected):
    assert second_epoch_decision(initial, mid, end)['continue'] is expected
