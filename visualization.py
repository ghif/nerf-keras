import os
import glob
import imageio.v2 as imageio
from tqdm import tqdm

import argparse
import json

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/lego_batch_h256.json")

args = parser.parse_args()

# Load config json
with open(args.config) as f:
    conf = json.load(f)

# Get config filename
config_filename = os.path.splitext(os.path.basename(args.config))[0]

img_dir = f"images/models/{config_filename}-best"
png_list = os.path.join(img_dir, "*.png")

filenames = glob.glob(png_list)
filenames = sorted(filenames)

images = []
for filename in tqdm(filenames):
    images.append(imageio.imread(filename))

# kargs = {"duration": 0.25}

# imageio.mimsave(f"{config_filename}_training.gif", images, "GIF", **kargs)

video_file = f"{config_filename}_training.mp4"
imageio.mimwrite(video_file, images, fps=30, quality=6, macro_block_size=None)