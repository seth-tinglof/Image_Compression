import numpy as np
from PIL import Image

FILENAME = "4.2.03.png"
RANK_APPROXIMATIONS = [1, 2, 5, 10, 25, 50, 100]


image_original = np.array(Image.open(FILENAME).convert("RGB"))
channels = []
for i in range(3):
    channels.append(np.linalg.svd(image_original[:, :, i]))

for approx in RANK_APPROXIMATIONS:
    image = np.zeros(image_original.shape, dtype=('uint8'))
    for i in range(3):
        u, s, v = channels[i]
        u, s, v = u.astype('float16'), s.astype('float16'), v.astype('float16') # correctly simulate 16bit precision
        temp = np.dot(u[:, :approx] * s[:approx], v[:approx, :])
        temp[temp < 0] = 0
        image[:, :, i] = temp
    Image.fromarray(image).save("output/simple_svd/rank_%d_approx.png" % approx)
