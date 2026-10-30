"""Finding people on a painted plate, so a stranger klein put there can be painted out before the subject is
pasted in. Florence-2 object detection, loaded from a local folder (no download)."""
from __future__ import annotations

import functools

import numpy as np

PERSON_WORDS = ("person", "man", "woman", "boy", "girl", "child", "people", "human", "hiker", "pedestrian")


def _shim_transformers() -> None:
    """Florence-2-base's bundled code predates transformers 5 (checked on the desktop with 5.15.1): its config
    reads ``forced_bos_token_id`` and its processor ``additional_special_tokens``, both gone. Put them back."""
    from transformers import PretrainedConfig
    for name in ("forced_bos_token_id", "forced_eos_token_id"):
        if not hasattr(PretrainedConfig, name):
            setattr(PretrainedConfig, name, None)
    tokenizers = []
    for name in ("RobertaTokenizer", "RobertaTokenizerFast"):
        try:
            tokenizers.append(getattr(__import__("transformers", fromlist=[name]), name))
        except (ImportError, AttributeError):
            pass
    for tokenizer in tokenizers:
        if not hasattr(tokenizer, "additional_special_tokens"):
            tokenizer.additional_special_tokens = property(lambda self: list(getattr(self, "extra_special_tokens",
                                                                                      None) or []))


FLORENCE_SIZE = 768  # its encoder takes square 768 px input; transformers 5's image processor no longer resizes


@functools.lru_cache(maxsize=1)
def _florence(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    _shim_transformers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, attn_implementation="eager",
                                                 torch_dtype=torch.float16
                                                 if device == "cuda" else torch.float32).to(device).eval()
    # transformers 5 no longer ties the shared embeddings into the encoder, decoder and head ("newly
    # initialized" in the load log), and the model then writes gibberish.
    lm = model.language_model
    for module in (lm.model.encoder.embed_tokens, lm.model.decoder.embed_tokens, lm.lm_head):
        module.weight = lm.model.shared.weight
    return model, AutoProcessor.from_pretrained(path, trust_remote_code=True), device


def _detect(model, processor, device, square, size, task: str, text: str = "") -> list:
    import torch
    inputs = processor(text=task + text, images=square, return_tensors="pt").to(device, model.dtype)
    with torch.no_grad():  # no cache: the bundled code indexes the cache as a tuple, which transformers 5 isn't
        ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             max_new_tokens=512, num_beams=3, do_sample=False, use_cache=False)
    found = processor.post_process_generation(processor.batch_decode(ids, skip_special_tokens=False)[0],
                                              task=task, image_size=size)[task]
    return list(zip(found.get("bboxes", []), found.get("labels", found.get("bboxes_labels", []))))


def find_people(image: np.ndarray, model_path: str) -> list[tuple[float, float, float, float]]:
    """Boxes (x0, y0, x1, y1, in pixels) of people Florence-2 finds in an RGB float image. Plain object
    detection missed the walker at f176 (r14k) and open-vocabulary "person" found it but once called the whole
    frame a person, so it is both, without boxes too big or too small to be someone on the plate."""
    from PIL import Image
    model, processor, device = _florence(model_path)
    picture = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    square = picture.resize((FLORENCE_SIZE, FLORENCE_SIZE), Image.BICUBIC)
    found = [box for box, label in _detect(model, processor, device, square, picture.size, "<OD>")
             if any(word in label.lower() for word in PERSON_WORDS)]
    found += [box for box, _ in _detect(model, processor, device, square, picture.size,
                                        "<OPEN_VOCABULARY_DETECTION>", "person")]
    return [tuple(box) for box in found if _person_sized(box, *picture.size)]


def _person_sized(box, cols: int, rows: int) -> bool:
    """Someone standing or walking on the plate: taller than wide, not most of the frame. OVD "person" once
    took the whole frame (r14k) and once a big oak (r17k, 504 x 559 px of 1360 x 768)."""
    width, height = box[2] - box[0], box[3] - box[1]
    return width < 0.25 * cols and height >= width and 0.02 * rows < height < 0.9 * rows


def people_mask(shape: tuple[int, int], boxes, grow: float = 0.25) -> np.ndarray:
    """A mask over each box, grown by ``grow`` of its size (more below, for the feet and shadow)."""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        mask[max(0, int(y0 - grow * h)):min(rows, int(y1 + 2 * grow * h) + 1),
             max(0, int(x0 - grow * w)):min(cols, int(x1 + grow * w) + 1)] = 1.0
    return mask


MAX_FILL_SHARE = 0.3  # a mask over more of the frame than this is a misdetection (r17k: an oak, 59%), left alone


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)-wide square, by cumulative sums along each axis; edges use the pixels there are."""
    out = image
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius + 1, radius)
        total = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        n = out.shape[axis]
        out = (np.take(total, range(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
               - np.take(total, range(0, n), axis=axis)) / (2 * radius + 1)
    return out


def fill_people(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The masked area filled smoothly from the ground around it, so the repaint that follows starts from
    empty ground instead of a person-shaped hole (repainted from noise in that hole, klein drew a new walker
    there: r16k f112/f176). Push-pull alone leaves blocks of averaged colour that the repaint kept (r17k), so
    the fill is then blurred over about a quarter of the hole's size, three times (near Gaussian)."""
    from geoblock_core.zoom import _fill
    hole = mask > 0.5
    if not hole.any() or hole.mean() > MAX_FILL_SHARE:
        return image
    filled = _fill(image, hole)
    rows, cols = np.nonzero(hole)
    radius = max(2, round(max(np.ptp(rows), np.ptp(cols)) / 8))
    y0, y1 = max(0, rows.min() - 3 * radius), min(hole.shape[0], rows.max() + 3 * radius + 1)
    x0, x1 = max(0, cols.min() - 3 * radius), min(hole.shape[1], cols.max() + 3 * radius + 1)
    soft = filled[y0:y1, x0:x1]
    for _ in range(3):
        soft = _box_blur(soft, radius)
    out = filled.copy()
    inside = hole[y0:y1, x0:x1]
    out[y0:y1, x0:x1][inside] = soft[inside]
    return out
