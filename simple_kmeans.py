import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

FILENAME = "4.2.03.png"
CLUSTERS = [2, 5, 10, 25, 50, 100]


image_original = np.array(Image.open(FILENAME).convert("RGB"))
w, h = image_original.shape[0], image_original.shape[1]
image_original = image_original.ravel().reshape(-1, 3)


for c in CLUSTERS:
    model = KMeans(c)
    clustering = model.fit_predict(image_original)
    image = np.zeros(image_original.shape, dtype='uint8')
    for i in range(len(image)):
        image[i] = model.cluster_centers_[clustering[i]]
    Image.fromarray(image.reshape((w, h, 3))).save("output/simple_kmeans/cluster_%d.png" % c)