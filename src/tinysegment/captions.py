"""Text overlays for the live camera preview."""
from functools import lru_cache
from math import ceil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _wrap(text, font, width):
    line = ''
    for word in text.split():
        candidate = f'{line} {word}' if line else word
        if font.getlength(candidate) <= width:
            line = candidate
            continue
        if line:
            yield line
        line = ''
        for char in word:
            if line and font.getlength(line + char) > width:
                yield line
                line = ''
            line += char
    if line:
        yield line


@lru_cache(maxsize=4)
def _caption_tile(text, width, height, size, is_prompt):
    """Rasterize only when the words or the preview dimensions change."""
    while True:
        font = ImageFont.load_default(size=size)
        pad = max(1, size // 4)
        lines = list(_wrap(text, font, max(1, width - 2 * pad)))
        line_height = sum(font.getmetrics())
        tile_height = line_height * len(lines) + 2 * pad
        if tile_height <= height or size == 1:
            break
        size = max(1, min(size - 1, int(size * (height / tile_height) ** 0.5)))
    tile_width = ceil(max(map(font.getlength, lines))) + 2 * pad
    tile = Image.new('RGB', (tile_width, tile_height), (24, 24, 24))
    draw = ImageDraw.Draw(tile)
    color = (96, 222, 255) if is_prompt else 'white'
    x = tile_width - pad if is_prompt else pad
    for row, line in enumerate(lines):
        draw.text((x, pad + row * line_height), line, font=font, fill=color,
                  anchor='ra' if is_prompt else 'la')
    tile.thumbnail((width, height))
    return np.asarray(tile)[:, :, ::-1]  # OpenCV preview frames are BGR.


def draw_captions(pixels, transcript, prompt):
    height, width = pixels.shape[:2]
    size = max(12, min(36, height // 32))
    margin = max(2, size // 2)
    column_width = width // 2 - 2 * margin
    available_height = min(height // 4, height - 2 * margin)
    if column_width <= 0 or available_height <= 0:
        return
    for text, is_prompt in ((transcript, False), (prompt, True)):
        if not text.strip():
            continue
        tile = _caption_tile(text, column_width, available_height, size, is_prompt)
        h, w = tile.shape[:2]
        x = width - margin - w if is_prompt else margin
        y = height - margin - h
        pixels[y:y + h, x:x + w] = tile
