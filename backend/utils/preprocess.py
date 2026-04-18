import numpy as np

def generate_positions(size, tile, step):
    pos = list(range(0, size - tile + 1, step))
    if pos[-1] + tile < size:
        pos.append(size - tile)
    return pos


def crop_tiles(img, tile_size=256, overlap=50):
    tiles = []
    positions = []

    step = tile_size - overlap

    C, H, W = img.shape

    xs = generate_positions(W, tile_size, step)
    ys = generate_positions(H, tile_size, step)

    for y in ys:
        for x in xs:
            tile = img[:, y:y+tile_size, x:x+tile_size]

            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions