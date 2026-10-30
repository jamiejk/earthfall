"""Phase-isolated text/connector context for an explicit Prompt Relay experiment.

Each phase is encoded with the shared motion instructions in its own text pass.
The LTX connector processes those passes separately, including its registers.
All resulting context positions belong to exactly one scheduled phase.
"""
from __future__ import annotations


def encode_phases(clip, global_prompt, beats):
    import torch
    tensors, lengths, metadata = [], [], None
    for beat in beats:
        result = clip.encode_from_tokens_scheduled(clip.tokenize(global_prompt+' '+beat['prompt']))
        if len(result) != 1 or not result[0][1].get('unprocessed_ltxav_embeds'):
            raise ValueError('Isolated phases require one unprocessed LTX context per phase')
        tensor, extra = result[0]
        tensors.append(tensor)
        lengths.append(tensor.shape[1])
        metadata = metadata or dict(extra)
    return [[torch.cat(tensors, dim=1), metadata]], lengths


def connect_phases(context, lengths, connector):
    """Never pass two phases through the connector together."""
    import torch
    if context.shape[1] != sum(lengths) or any(n < 1 for n in lengths):
        raise ValueError('Phase context boundary mismatch')
    outputs = [connector(chunk, unprocessed=True) for chunk in context.split(lengths, dim=1)]
    ranges, cursor = [], 0
    for out in outputs:
        if out.shape[0] != context.shape[0] or out.shape[1] < 1:
            raise ValueError('Unexpected connector output')
        ranges.append((cursor, cursor+out.shape[1]))
        cursor += out.shape[1]
    return torch.cat(outputs, dim=1), ranges


def validate_partition(ranges, key_count, phases):
    cursor = 0
    if len(ranges) != phases:
        raise ValueError('Missing phase context')
    for lo, hi in ranges:
        if lo != cursor or hi <= lo:
            raise ValueError('Unassigned or overlapping phase context')
        cursor = hi
    if cursor != key_count:
        raise ValueError('Unassigned connector registers')
