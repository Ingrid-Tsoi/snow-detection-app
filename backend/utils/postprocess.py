import numpy as np

def stitch_tiles(tiles, positions, full_shape, tile_size=256):
    _, H, W = full_shape

    output = np.zeros((H, W), dtype=np.float32)
    count  = np.zeros((H, W), dtype=np.float32)

    for tile, (y, x) in zip(tiles, positions):
        output[y:y+tile_size, x:x+tile_size] += tile
        count[y:y+tile_size, x:x+tile_size] += 1

    count[count == 0] = 1
    output = output / count

    return (output > 0.5).astype(np.uint8)