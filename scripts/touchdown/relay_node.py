"""Isolated ComfyUI node: pinned Kijai Prompt Relay with explicit guide times.

Copied into the experiment's extra custom-node directory by relay_experiment.
Upstream patches.py/prompt_relay.py and their hashes live alongside this file.
"""
import json
import logging
from pathlib import Path
import types

from .patches import detect_model_type
from .prompt_relay import get_raw_tokenizer, map_token_indices
from touchdown.relay_timing import query_positions, temporal_mask, direct_attention
from touchdown.relay_context import encode_phases, connect_phases, validate_partition

log = logging.getLogger(__name__)

# A guide pinned below full strength makes LTX build a per-group additive self-attention
# mask (comfy/ldm/lightricks/model.py::_attention_with_guide_mask). On this stack cuDNN
# cannot build a plan for that masked SDPA and the whole run dies with "No valid execution
# plans built". Restrict only MASKED CUDA calls to PyTorch's memory-efficient kernel; math
# is not offered because a 33,120-token video self-attention does not fit in 24 GB. Unmasked
# attention keeps ComfyUI's own backend choice, so full-strength guides are unaffected.
def _masked_sdpa_kernel_fix():
    import torch
    import torch.nn.functional as F
    import comfy.ops as ops
    original = ops.scaled_dot_product_attention
    if getattr(original, "_touchdown_masked_kernel", False):
        return
    def patched(q, k, v, *args, **kwargs):
        mask = kwargs.get("attn_mask")
        if mask is not None and torch.is_tensor(q) and q.is_cuda:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            if kwargs.get("enable_gqa") and q.shape[-3] != k.shape[-3]:
                kwargs.pop("enable_gqa")
                repeat = q.shape[-3] // k.shape[-3]
                k = k.repeat_interleave(repeat, dim=-3)
                v = v.repeat_interleave(repeat, dim=-3)
            # ComfyUI's own wrapper re-enters its backend-priority context, so the
            # masked call has to go straight to the torch kernel rather than nest.
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, *args, **kwargs)
        return original(q, k, v, *args, **kwargs)
    patched._touchdown_masked_kernel = True
    ops.scaled_dot_product_attention = patched


_masked_sdpa_kernel_fix()


class TouchdownPromptRelay:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "latent": ("LATENT",),
                "timing_json": ("STRING",), "guide_frames": ("STRING",), "audit_path": ("STRING",),
                "enabled": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "encode"
    CATEGORY = "touchdown/experimental"

    def encode(self, model, clip, latent, timing_json, guide_frames, audit_path, enabled=True):
        timing, guides = json.loads(timing_json), json.loads(guide_frames)
        arch, patch_size, stride = detect_model_type(model)
        if arch != "ltx" or stride != 8:
            raise ValueError("This photographic experiment requires the LTX 8-frame temporal grid")
        shape = latent["samples"].shape
        frames, tpf = shape[2], shape[3]*shape[4]
        beats = timing["generation_beats"]
        strategy = timing.get('context_strategy', 'token-spans-v1')
        if strategy not in ('token-spans-v1', 'isolated-phases-v1'):
            raise ValueError('Unknown context experiment')
        raw = get_raw_tokenizer(clip)
        text, ranges = map_token_indices(raw, timing["global_prompt"], [b["prompt"] for b in beats])
        if text != timing["single_prompt"]:
            raise ValueError("Baseline and relay text differ")
        tokens = clip.tokenize(text)
        raw_ids = raw(text)["input_ids"]
        if len(tokens) != 1:
            raise ValueError("Unexpected LTX tokenizer streams")
        rows = next(iter(tokens.values()))
        if len(rows) != 1:
            raise ValueError("Unexpected LTX text chunks")
        ids = [t[0] for t in rows[0]]
        while ids and ids[0] == 0:
            ids.pop(0)
        # Gemma 4's raw tokenizer omits BOS; Comfy's wrapper adds it.
        # Gemma 3 raw tokenisation may already include it. Verify, don't guess.
        offset = len(ids)-len(raw_ids)
        if offset not in (0, 1) or ids[offset:] != raw_ids:
            raise ValueError("Cannot align Prompt Relay token positions with the actual LTX encoder")
        ranges = [(lo+offset, hi+offset) for lo, hi in ranges]
        positive = clip.encode_from_tokens_scheduled(tokens)
        encoded_count = positive[0][0].shape[1]
        if encoded_count != len(ids):
            raise ValueError("LTX encoder changed text positions; refusing a misleading timeline")
        phase_lengths = None
        if strategy == 'isolated-phases-v1':
            positive, phase_lengths = encode_phases(clip, timing['global_prompt'], beats)
            # Assigned only after each independent connector pass has completed.
            ranges = []
        audit = dict(architecture=arch, temporal_stride=stride, latent_frames=frames, temporal_weighting=enabled,
                     attention_backend="PyTorch efficient SDPA (cuDNN disabled for this cross-attention)",
                     tokens_per_frame=tpf, token_offset=offset, text_tokens=encoded_count,
                     token_ranges=ranges, guide_pixel_frames=guides,
                     generation_beats=beats, attention_calls=[], context_strategy=strategy,
                     phase_input_lengths=phase_lengths, connector_calls=[])
        audit['raw_context_positions'] = positive[0][0].shape[1]
        target = Path(audit_path); target.parent.mkdir(parents=True, exist_ok=True)
        def save():
            target.write_text(json.dumps(audit, indent=2))
        save()
        cache = {}
        patched = model.clone()
        if phase_lengths:
            original_connector = patched.get_model_object('diffusion_model.preprocess_text_embeds')
            def phase_connector(_self, context, unprocessed=False):
                nonlocal ranges
                if not unprocessed or context.shape[1] != sum(phase_lengths):
                    # Negative conditioning, if evaluated, retains its native connector.
                    return original_connector(context, unprocessed=unprocessed)
                result, assigned = connect_phases(context, phase_lengths, original_connector)
                validate_partition(assigned, result.shape[1], len(beats))
                ranges = assigned
                audit['connector_calls'].append(dict(input_lengths=phase_lengths,
                    output_ranges=ranges, output_positions=result.shape[1], unrestricted_positions=0,
                    independent_text_encoding=True, independent_connector_calls=len(phase_lengths)))
                audit['token_ranges'] = ranges
                save()
                return result
            module = patched.get_model_object('diffusion_model')
            patched.add_object_patch('diffusion_model.preprocess_text_embeds', types.MethodType(phase_connector, module))
        def stage_tokens_per_frame(Lq):
            """Tokens per latent plane for the stage that is actually sampling.

            A spatial upscale multiplies this ratio and changes nothing else about the clock,
            so the mask has to be rebuilt at the stage-2 ratio instead of refusing to run.
            """
            planes = frames + len(guides)
            if planes < 1 or Lq % planes:
                raise ValueError("Unrecognised Prompt Relay query geometry")
            value = Lq // planes
            if value < tpf or value % tpf:
                raise ValueError("Prompt Relay query count is not a whole multiple of the stage-1 frame")
            return value

        def mask_fn(Lq, Lk, dtype, device, transformer_options):
            modes = transformer_options.get("cond_or_uncond", [])
            if 1 in modes and 0 not in modes:
                return None
            if phase_lengths:
                validate_partition(ranges, Lk, len(beats))
            if Lk < max(hi for _, hi in ranges):
                raise ValueError("Prompt Relay received a different text context")
            key = (Lq, Lk, str(dtype), str(device))
            if key not in cache:
                stage_tpf = stage_tokens_per_frame(Lq)
                times = query_positions(frames, stage_tpf, Lq, guides)
                cache[key] = temporal_mask(times, stage_tpf, ranges, beats, Lk,
                                            timing["epsilon"], dtype, device)
                if not enabled:
                    cache[key].zero_()
                audit["attention_calls"].append(dict(query_tokens=Lq, text_tokens=Lk,
                    video_latent_frames=frames, appended_guide_planes=len(guides),
                    tokens_per_frame=stage_tpf, stage_tokens_per_frame=stage_tpf,
                    video_pixel_frame_centres=times[:frames],
                    appended_guide_pixel_frame_centres=times[frames:], clock_rescaled=False,
                    unassigned_context_positions=Lk-sum(hi-lo for lo,hi in ranges)))
                save()
                log.info("Touchdown Prompt Relay: %d video planes + %d guide planes; original clock retained", frames, len(guides))
            return cache[key]
        # cuDNN cannot execute this arbitrary additive mask on the installed
        # stack. Restrict only the patched text cross-attention to efficient
        # SDPA, for BOTH variants. No process-wide backend setting is changed.
        def efficient_wrapper(underlying):
            def forward(_self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
                if context is not None:
                    relay_mask=mask_fn(x.shape[1],context.shape[1],x.dtype,x.device,transformer_options)
                    if relay_mask is not None:
                        mask=relay_mask if mask is None else mask+relay_mask
                options={**transformer_options,"optimized_attention_override":direct_attention}
                return underlying(x,context=context,mask=mask,pe=pe,k_pe=k_pe,transformer_options=options)
            return forward
        for i, block in enumerate(patched.get_model_object("diffusion_model").transformer_blocks):
            for name in ("attn2", "audio_attn2"):
                module = getattr(block, name, None)
                if module is not None:
                    key = f"diffusion_model.transformer_blocks.{i}.{name}.forward"
                    patched.add_object_patch(key, types.MethodType(efficient_wrapper(patched.get_model_object(key)), module))
        return patched, positive


class TouchdownRelayProbe:
    """Run actual encoders/connectors and an independence check without diffusion."""
    INPUT_TYPES = TouchdownPromptRelay.INPUT_TYPES
    RETURN_TYPES = ()
    FUNCTION = 'probe'
    CATEGORY = 'touchdown/experimental'
    OUTPUT_NODE = True

    def probe(self, **kwargs):
        import torch
        import comfy.model_management as management
        patched, positive = TouchdownPromptRelay().encode(**kwargs)
        management.load_models_gpu([patched])
        dtype = patched.model.get_dtype_inference()
        context = positive[0][0].to(device=patched.load_device, dtype=dtype)
        hook = patched.get_model_object('diffusion_model.preprocess_text_embeds')
        with torch.inference_mode():
            out = hook(context, unprocessed=True)
            path = Path(kwargs['audit_path'])
            audit = json.loads(path.read_text())
            if audit['context_strategy'] == 'isolated-phases-v1':
                ranges = audit['token_ranges']
                validate_partition(ranges, out.shape[1], len(audit['generation_beats']))
                changed = context.clone()
                changed[:, -audit['phase_input_lengths'][-1]:] = 0
                altered = hook(changed, unprocessed=True)
                untouched = float((out[:, :ranges[-1][0]]-altered[:, :ranges[-1][0]]).abs().max())
                affected = float((out[:, ranges[-1][0]:]-altered[:, ranges[-1][0]:]).abs().max())
                if untouched != 0 or not affected > 0:
                    raise ValueError('Actual LTX connector failed phase independence')
                audit['independence_probe'] = dict(unchanged_phase_max_error=untouched,
                    changed_phase_max_difference=affected, status='pass')
            audit['probe'] = dict(status='pass', output_positions=out.shape[1], diffusion_steps=0)
            path.write_text(json.dumps(audit, indent=2))
        return {'ui': {'text': [str(path)]}}


NODE_CLASS_MAPPINGS = {"TouchdownPromptRelay": TouchdownPromptRelay, 'TouchdownRelayProbe': TouchdownRelayProbe}
