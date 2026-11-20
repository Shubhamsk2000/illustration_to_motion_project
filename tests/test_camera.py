import pytest

from motion_studio.camera import ALL_MOVES, LOOP_MOVES, ONE_WAY_MOVES, auto_move, camera_at


@pytest.mark.parametrize("move", ALL_MOVES)
def test_zoom_never_below_one(move):
    for i in range(101):
        assert camera_at(move, i / 100, strength=2.0)[0] >= 1.0


def test_push_in_and_pull_out_are_opposites():
    assert camera_at("push_in", 0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert camera_at("push_in", 1.0)[0] > 1.05
    assert camera_at("pull_out", 0.0) == pytest.approx(camera_at("push_in", 1.0))
    assert camera_at("pull_out", 1.0) == pytest.approx((1.0, 0.0, 0.0))


def test_pans_travel_in_opposite_directions():
    assert camera_at("pan_right", 1.0)[1] < camera_at("pan_right", 0.0)[1]
    assert camera_at("pan_left", 1.0)[1] > camera_at("pan_left", 0.0)[1]


@pytest.mark.parametrize("move", LOOP_MOVES)
def test_loop_moves_end_where_they_start(move):
    assert camera_at(move, 1.0) == pytest.approx(camera_at(move, 0.0))


def test_one_way_moves_start_and_stop_gently():
    for move in ONE_WAY_MOVES:
        a, b = camera_at(move, 0.0), camera_at(move, 0.01)
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-3


def test_auto_move_varies_between_neighbours():
    seq = [auto_move(i) for i in range(6)]
    assert all(x != y for x, y in zip(seq, seq[1:]))


def test_unknown_move_raises():
    with pytest.raises(ValueError):
        camera_at("barrel_roll", 0.5)
