import os
from PIL import Image
import sys
root_dir = sys.argv[1]

images = os.listdir(root_dir)

images = list(map(lambda p : os.path.join(root_dir, p), images))

colors = set()

for image in images:
    img = Image.open(image)
    l = img.getcolors()
    for color in l:
        colors.add(color[1])

colors = list(colors)

print(len(images))
print(len(colors))
print(colors)
