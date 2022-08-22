import cv2
import matplotlib.pyplot as plt

from utils import pseudo2gray, Model, Regions

rgb = cv2.imread("../images/FLIR2220_rgb_image.jpg")

pseudocolor = cv2.imread("../images/FLIR2220.jpg")
temp_range = (26.0, 40.0)
temp_map, _ = pseudo2gray(pseudocolor, temp_range)

coco = Model("coco")
mpi = Model("mpi")

coco.detect_points(rgb)
mpi.detect_points(rgb)

coco.display(rgb)

regions = Regions(coco, mpi)
regions.print_stats(temp_map)
copy = regions.display(pseudocolor)

plt.figure()
plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))

cv2.imwrite("result.png", copy)

regions.display_histograms(temp_map)