from geoblock_core.motion_blur import filter_graph, weight_expr, windows


def test_windows_follow_the_shot_and_match_the_r46_recipe():
    spans = windows(241, 30.9, 108)
    assert spans == {"light": (24, 116), "strong": (80, 104)}
    later = windows(241, 40.5, 120)
    assert later["light"][0] > spans["light"][0] and later["strong"] == (92, 116)


def test_filter_graph_blends_two_shutters_on_the_frame_number():
    graph = filter_graph(24.0, windows(241, 30.9, 108))
    assert "minterpolate=fps=96" in graph and "tmix=frames=2" in graph and "tmix=frames=4" in graph
    assert graph.count("blend=all_expr") == 2 and graph.endswith("[out]")
    assert weight_expr((80, 104)) in graph
