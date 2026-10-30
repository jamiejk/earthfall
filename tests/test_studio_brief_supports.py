"""Which surface labels a brief accepts.

The Teotihuacan run was held twice for standing on natural_rock when the grounding model had labelled
the same surface ground_paving and described it as rough grey stone. A good take must not be lost to
that word, and the acceptance must not widen so far that a person standing on the wrong surface
passes.
"""
import pytest

from touchdown.studio_brief import ADJACENT_SUPPORT, GROUND, allowed_supports


def test_the_labelled_surface_is_always_accepted_first():
    for surface in ("grass", "ground_paving", "natural_rock", "sand"):
        assert allowed_supports(surface)[0] == surface


def test_hard_dry_stone_is_interchangeable_either_way():
    assert set(allowed_supports("ground_paving")) == {"ground_paving", "natural_rock"}
    assert {"natural_rock", "ground_paving"} <= set(allowed_supports("natural_rock"))


def test_sand_may_read_as_rock_but_rock_does_not_excuse_sand():
    assert "natural_rock" in allowed_supports("sand")
    assert "sand" not in allowed_supports("natural_rock")


def test_a_person_on_grass_is_not_accepted_for_a_paved_landmark():
    assert "grass" not in allowed_supports("ground_paving")


def test_grassy_ground_accepts_its_paths_and_rock():
    # Neuschwanstein (rock brief, grassy slope) and Hohenzollern (grass brief, gravel path) were held over the word.
    assert "grass" in allowed_supports("natural_rock")
    assert allowed_supports("grass") == ["grass", "ground_paving", "natural_rock"]


def test_the_table_covers_the_whole_taxonomy():
    for surface in ("grass", "ground_paving", "natural_rock", "sand"):
        assert surface in ADJACENT_SUPPORT
        assert set(allowed_supports(surface)) <= set(GROUND.__args__)
