"""The one-shot LLM helper's offline pieces: JSON extraction and image encoding."""
from PIL import Image

from autopc.agent_brain.llm import MAX_IMAGE_EDGE, extract_json, image_block


def test_extract_json_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_with_prose():
    text = 'Sure, here it is:\n```json\n{"overall": "pass", "n": 2}\n```\nDone.'
    assert extract_json(text) == {"overall": "pass", "n": 2}


def test_extract_json_embedded_in_prose():
    assert extract_json('Result: {"ok": true} -- end') == {"ok": True}


def test_extract_json_rejects_garbage():
    assert extract_json("no json here") is None
    assert extract_json("") is None
    assert extract_json("[1, 2, 3]") is None


def test_image_block_downscales_long_edge():
    big = Image.new("RGB", (3000, 1000), "white")
    block = image_block(big)
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] in ("image/webp", "image/png")
    import base64, io
    raw = base64.b64decode(block["source"]["data"])
    w, h = Image.open(io.BytesIO(raw)).size
    assert max(w, h) <= MAX_IMAGE_EDGE


def test_image_block_keeps_small_images_native():
    import base64, io
    small = Image.new("RGB", (800, 600), "white")
    raw = base64.b64decode(image_block(small)["source"]["data"])
    assert Image.open(io.BytesIO(raw)).size == (800, 600)
