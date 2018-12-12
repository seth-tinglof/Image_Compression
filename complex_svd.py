import numpy as np
from PIL import Image

FILENAME = "4.2.03.png"
RANK_APPROXIMATIONS = [1, 2, 3, 4, 5]
BLOCK = (25, 25)


image_original = np.array(Image.open(FILENAME).convert("RGB"))
block_svds = []
for i in range(0, image_original.shape[0], BLOCK[0]):
    for j in range(0, image_original.shape[1], BLOCK[1]):
        channels = []
        for k in range(3):
            channels.append(np.linalg.svd(image_original[i:i+BLOCK[0], j:j+BLOCK[1], k]))
        block_svds.append(channels)

for approx in RANK_APPROXIMATIONS:
    image = np.zeros(image_original.shape, dtype='uint8')
    for i in range(len(block_svds)):
        y = (i % (image_original.shape[1] // BLOCK[1])) * BLOCK[1]
        x = (i // (image_original.shape[1] // BLOCK[1])) * BLOCK[0]
        for j in range(3):
            u, s, v = block_svds[i][j]
            u, s, v = u.astype('float16'), s.astype('float16'), v.astype('float16') # correctly simulate 16bit precision
            temp = np.dot(u[:, :approx] * s[:approx], v[:approx, :])
            temp[temp < 0] = 0
            temp[temp > 255] = 255
            image[x:x+BLOCK[0], y:y+BLOCK[1], j] = temp
    Image.fromarray(image).save("output/complex_svd/rank_%d_block_%d_approx.png" % (approx, BLOCK[0]))