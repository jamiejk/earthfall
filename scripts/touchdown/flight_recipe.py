"""Shared app/recovery recipe; product clock and regional instructions live here."""
from touchdown import flight_contract, split_flight as flight


def configuration(source_cfg):
    cfg=flight_contract.configure(source_cfg)
    w=flight.words(cfg)
    cfg['prompts']['high']=(
        f"One real aerial photograph from 30,000 FEET (9,144 METRES) ABOVE GROUND over {w['region']}, looking DOWN "
        f"at 45 degrees, with SKY AND HORIZON ENTIRELY OUTSIDE THE PICTURE. Show tens of kilometres of connected regional terrain and water: {w['pattern']}. "
        f"Natural {w['palette']} colours, {w['light']}. Broad connected landforms and tiny settlement "
        "patterns extend to every edge INCLUDING THE TOP EDGE. Include regional lake basins and shorelines "
        "where appropriate to this region. This is the clear regional view below the high cloud deck; no opaque "
        "foreground cloud, no eye-level horizon. The local landmark is microscopic and no individual facade "
        "can be read. People cannot be resolved. Natural photographic atmospheric depth and irregular detail. "
        "No sky, no horizon line, no curved Earth, no planetary limb, no space. No enlarged castle, no giant "
        "monument, no model village, no repeated synthetic mountains, no illustration, no CGI, no inset or text. "
        "Rectilinear 35 mm lens, fixed focal length, no fisheye or barrel distortion.")
    flight.options(cfg)
    return cfg
