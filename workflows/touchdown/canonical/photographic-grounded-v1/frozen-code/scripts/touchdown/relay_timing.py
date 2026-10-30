"""Explicit delivery beats and guide-aware timing for photographic LTX trials."""
from __future__ import annotations

import math

FPS = 24
FRAMES = 193
DELIVERY_FRAMES = 192
BOUNDS = (0, 29, 58, 108, 156, 192)
NAMES = ("orbit", "clouds", "high_aerial", "approach", "settle")


def schedule(cfg, opening_handles=0):
    """Prompts describe the actual reverse-generation action, not a numbered wish list."""
    if opening_handles not in (0, 48):
        raise ValueError("Use zero or 48 extra orbital frames")
    # Concrete scene nouns belong only to the final approach/portrait. Global
    # conditioning must not put a person or a landmark onto the planet.
    global_prompt = (
        "One uninterrupted live-action photographic camera move. Natural muted green-brown land, "
        "soft natural daylight, consistent illumination and exposure. Real irregular surface detail, "
        "atmospheric depth and fixed focal length. Maintain one heading and continuous movement. "
        "Physical perspective and parallax, no illustration, no CGI, no cuts, wipes or dissolves."
    )
    local = [
        "The retreat finishes close above the broad curved horizon of Earth. The Earth fills the picture, "
        "a thin blue atmospheric limb and a narrow band of black space above. Thick rounded white weather "
        "systems cover broad regional terrain. Keep physically moving above cloud tops. Remain close to "
        "the planet, never reveal an isolated whole globe. No individual structures or people are visible.",
        "Continue ascending rapidly into a deep substantial cloud layer. Billowing rounded cloud masses "
        "with illuminated tops and shaded recesses sweep close past the lens. The clouds fill and briefly "
        "obscure the whole view as the camera travels through them, then the camera breaks out above "
        "the cloud tops. Always move through real depth; no flat transparent veil or drawn streaks.",
        "The camera retreats upward at approximately 45 degrees, already HIGH above the ground. "
        "Several kilometres of irregular agricultural fields, sparse hedgerows, small woods and narrow "
        "country roads extend in every direction. Any local landmark is microscopic and people are "
        "unresolvable. Keep the region clearly visible BELOW the clouds throughout this interval. "
        "Only at the END of this interval reach the underside of the cloud layer and begin entering it.",
        f"Continue retreating BACKWARD AND UPWARD over the SAME connected ground from {cfg['place']}. "
        "The person shrinks into the setting, then becomes a tiny unresolvable speck as the camera "
        "gains altitude. The familiar landmark stays in its wider setting, never on a new platform. "
        "By the END of this interval reach a true broad high oblique aerial, about two kilometres "
        "above ground. Follow the open ground into the surrounding fields, no hovering or pause.",
        "Begin with the exact supplied moving photographic portrait, already moving. Gently retreat "
        "away from the same person along the supplied reversed camera motion, accelerating smoothly "
        "toward the next interval. Preserve the supplied person, clothes, natural light and connected "
        "ground, with natural body and fabric motion. Never replace the moving portrait with a still.",
    ]
    forward = []
    for i, name in enumerate(NAMES):
        forward.append(dict(name=name, start_frame=BOUNDS[i], end_frame=BOUNDS[i+1],
                            start_seconds=BOUNDS[i]/FPS, end_seconds=BOUNDS[i+1]/FPS,
                            reverse_prompt=local[i]))
    # Keep all 193 source frames on the generation clock. Generation frame 0
    # is the extra portrait endpoint, trimmed after reversal. All surviving
    # frame intervals then map exactly back to the delivery boundaries.
    generation = []
    cursor = 0
    for beat in reversed(forward):
        length = (beat["end_frame"]-beat["start_frame"] + (beat["name"] == "settle")
                  + (opening_handles if beat["name"] == "orbit" else 0))
        generation.append(dict(name=beat["name"], start_frame=cursor, end_frame=cursor+length,
                               prompt=beat["reverse_prompt"]))
        cursor += length
    assert cursor == FRAMES+opening_handles
    return dict(version="earth-zoom-relay-v1", fps=FPS, source_frames=FRAMES+opening_handles, opening_handle_frames=opening_handles,
                delivery_frames=DELIVERY_FRAMES, global_prompt=global_prompt,
                forward_beats=forward, generation_beats=generation,
                single_prompt=global_prompt+" "+" ".join(b["prompt"] for b in generation),
                epsilon=.1, temporal_stride=8, native_portrait_frames=73)


def query_positions(latent_frames, tokens_per_frame, query_count, guide_frames):
    """Actual pixel-time centres, with appended references at their own times.

    Never spread T+K guide frames over the original T-frame video clock.
    The first causal latent covers frame 0; later latents cover eight frames.
    Each guide entry is one full-size latent plane in append order.
    """
    if tokens_per_frame < 1 or query_count % tokens_per_frame:
        raise ValueError("Unrecognised Prompt Relay query geometry")
    planes = query_count//tokens_per_frame
    if planes != latent_frames+len(guide_frames):
        raise ValueError("Guide token count changed; refusing to shift the prompt timeline")
    times = [0.0]+[8*i-3.5 for i in range(1, latent_frames)]
    return times+list(guide_frames)


def temporal_mask(positions, tokens_per_frame, token_ranges, beats, key_count, epsilon,
                  dtype, device):
    """Prompt Relay's temporal cross-attention penalty on explicit pixel times."""
    import torch
    if not 0 < epsilon < 1:
        raise ValueError("Prompt Relay epsilon must lie between zero and one")
    q = torch.tensor(positions, device=device, dtype=torch.float32).repeat_interleave(tokens_per_frame)
    mask = torch.zeros((len(q), key_count), device=device, dtype=dtype)
    # One latent frame of smooth transition. Inside each requested interval
    # its text is unpenalised; outside it decays, reaching epsilon 8 frames out.
    sigma = 8/math.sqrt(2*math.log(1/epsilon))
    for (lo, hi), beat in zip(token_ranges, beats, strict=True):
        if not 0 <= lo < hi <= key_count:
            raise ValueError("Temporal prompt tokens do not match encoded text")
        distance = (beat["start_frame"]-q).clamp(min=0)+(q-beat["end_frame"]).clamp(min=0)
        cost = -(distance**2/(2*sigma*sigma)).clamp(max=10000)
        mask[:, lo:hi] = cost[:, None].to(dtype)
    return mask


def direct_attention(_func, q, k, v, heads, mask=None, **kwargs):
    """Masked SDPA without Comfy's inner wrapper re-enabling cuDNN.

    LTX text cross-attention supplies flat B×tokens×(heads*dim) tensors.
    Keep this override local to those calls; other model attention is unchanged.
    """
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if kwargs.get("skip_reshape") or kwargs.get("skip_output_reshape") or kwargs.get("enable_gqa"):
        raise ValueError("Unexpected LTX text attention layout")
    b, count, width=q.shape
    dim=width//heads
    q,k,v=[x.reshape(b,-1,heads,dim).transpose(1,2) for x in (q,k,v)]
    if mask is not None:
        while mask.ndim < 4:
            mask=mask.unsqueeze(0)
        mask=mask.to(dtype=q.dtype)
    backend=SDPBackend.EFFICIENT_ATTENTION if q.is_cuda else SDPBackend.MATH
    with sdpa_kernel([backend]):
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,
            dropout_p=0.0,is_causal=False,**({"scale":kwargs["scale"]} if "scale" in kwargs else {}))
    return out.transpose(1,2).reshape(b,count,width)


def wire_relay(graph, timing, audit_path, guide_frames, enabled=True):
    """Use the same text/content as the control; only temporal weighting differs."""
    graph["305"]["inputs"]["text"] = timing["single_prompt"]
    graph["380"] = dict(class_type="TouchdownPromptRelay", inputs=dict(
        model=["300", 0], clip=["304", 0], latent=["308", 0],
        timing_json=__import__("json").dumps(timing), guide_frames=__import__("json").dumps(guide_frames),
        audit_path=str(audit_path), enabled=enabled))
    graph["307"]["inputs"]["positive"] = ["380", 1]
    for node in graph.values():
        if node["class_type"] == "CFGGuider":
            node["inputs"]["model"] = ["380", 0]
    return graph
