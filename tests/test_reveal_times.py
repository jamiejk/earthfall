"""Sample times for the cloud-to-ground reveal.

The build died at pacing with "Supply 1-12 finite nonnegative sample times" on the first Teotihuacan
take: a 241-frame flight against a 73-frame portrait window asks for 13 samples of a limit of 12. The
sample count is a checker constraint, not a reason to lose a film, so it caps and widens the spacing.
"""
from touchdown.studio_build import reveal_frame_index, reveal_times


def seen(*pairs):
    """Answers as (stage, terrain_clear)."""
    return [{"stage": stage, "terrain_clear": clear} for stage, clear in pairs]


def test_a_long_flight_against_a_short_portrait_stays_within_the_limit():
    times = reveal_times(241, 73)                       # the run that failed
    assert len(times) <= 12
    assert times == sorted(set(times))


def test_every_sampling_is_accepted_by_the_checker():
    for count, near in [(241, 73), (192, 96), (97, 49), (240, 24), (300, 100), (48, 24)]:
        times = reveal_times(count, near)
        assert 1 <= len(times) <= 12, (count, near, len(times))
        assert all(t >= 0 for t in times)
        assert times == sorted(set(times))


def test_the_usual_case_keeps_half_second_spacing():
    """Scenes that already pass must not change behaviour."""
    assert reveal_times(192, 96, fps=24) == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def test_capping_still_covers_the_whole_window():
    count, near = 241, 73
    times = reveal_times(count, near)
    window_seconds = (count - near - 1) / 24
    assert times[-1] == round(window_seconds, 3)
    assert times[0] > 0


def test_a_portrait_that_leaves_no_window_returns_one_sample():
    assert len(reveal_times(100, 99)) == 1


def test_two_clear_readings_establish_the_reveal():
    assert reveal_frame_index(seen(("aerial", True), ("aerial", True))) == 0
    assert reveal_frame_index(seen(("space", False), ("aerial", True), ("approach", True))) == 1


def test_one_misread_does_not_lose_a_clean_reveal():
    """The Teotihuacan take: clear at 5.8 s, misread at 6.4 s, clear at 7.0 s."""
    answers = seen(("space", False), ("aerial", True), ("portrait", False), ("portrait", True))
    assert reveal_frame_index(answers) == 1


def test_a_single_glimpse_of_ground_is_still_refused():
    assert reveal_frame_index(seen(("space", False), ("aerial", True), ("cloud", False),
                                   ("space", False))) is None


def test_a_cloud_frame_never_counts_even_when_marked_clear():
    assert reveal_frame_index(seen(("cloud", True), ("cloud", True))) is None


def test_no_ground_at_all_is_refused():
    assert reveal_frame_index(seen(("space", False), ("space", False))) is None
    assert reveal_frame_index([]) is None
