import cv2 as cv
import numpy as np
from PIL import Image
import os

black = np.full((256,256), 0.)
# cv.imshow('white image', white)
out_dir = '/share/wandell/data/reith/circle_fun/shapes/'
os.makedirs(out_dir, exist_ok=True)
# create logspace of radians, use of 128.001, as with 128, the function creates an array of 20 logarithmically spaced
# numbers
radii = np.unique(np.geomspace(1, 128.001, 5).astype(np.int))
for rad in radii:
    painting = cv.circle(black, center=(128,128), radius=rad, color=255, thickness=-1)
    img = Image.fromarray(painting)
    out_name = os.path.join(out_dir, f"five_white_circle_rad_{rad}.bmp")
    img.convert("L").save(out_name)
print('nice')
