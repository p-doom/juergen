from __future__ import annotations

import io

from PIL import Image

from rl.target_box.runtime import _annotate, _box, _cursor


def test_scene_is_reproducible_and_starts_outside() -> None:
    scene = _box(index=7, seed=3, screen=(1920, 1080), size=(150, 150), margin=40)
    assert scene == _box(
        index=7, seed=3, screen=(1920, 1080), size=(150, 150), margin=40
    )
    cursor = _cursor(index=7, seed=3, screen=(1920, 1080), box=scene)
    assert not (scene[0] <= cursor[0] < scene[2] and scene[1] <= cursor[1] < scene[3])


def test_annotation_keeps_the_jpeg_q92_wire_shape() -> None:
    source = io.BytesIO()
    Image.new("RGB", (1920, 1080), (20, 30, 40)).save(
        source, format="JPEG", quality=92, subsampling=2, optimize=False
    )
    encoded = _annotate(source.getvalue(), (100, 100, 250, 250))

    assert encoded.startswith(b"\xff\xd8") and encoded.endswith(b"\xff\xd9")
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.mode == "RGB"
        assert image.size == (1920, 1080)
