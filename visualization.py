import os
import glob
import imageio.v2 as imageio
from tqdm import tqdm

img_dir = "images/models/tinynerf-keras-best-256"
png_list = os.path.join(img_dir, "*.png")

filenames = glob.glob(png_list)
filenames = sorted(filenames)

images = []
for filename in tqdm(filenames):
    images.append(imageio.imread(filename))

kargs = {"duration": 0.25}

imageio.mimsave("training.gif", images, "GIF", **kargs)