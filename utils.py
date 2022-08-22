import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def pseudo2gray(image, temp_range):

    # Extract color bar
    bar = image[29:212,308:312,:]
    bar_256 = cv2.resize(bar, (2, 256))
    map = np.mean(bar_256, axis=1)

    M, N, C = image.shape

    gray = np.zeros((M, N), dtype=np.uint8)

    for m in range(M):
        for n in range(N):
            px = np.squeeze(image[m,n,:])
            mapdist = np.sqrt(np.sum(np.square(map - px), axis=1))
            i = np.argmin(mapdist)
            gray[m, n] = 255 - i

    median = cv2.medianBlur(gray, 5)

    gray2temp = np.linspace(temp_range[0], temp_range[1], 256)
    temp_map = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            temp_map[m, n] = gray2temp[gray[m, n]]

    return temp_map, median


class Regions:

    def __init__(self, coco_model, mpi_model, offset=(1.475, -48, -30)):
        
        self.polygons = []

        rf = offset[0]
        hoffset = offset[1]
        voffset = offset[2]

        neck = (int((coco_model.points[1][0] + mpi_model.points[1][0]) / 2 / rf) + hoffset, int(mpi_model.points[1][1] / rf) + voffset)
        up_back = (int(coco_model.points[1][0] / rf) + hoffset, int(coco_model.points[1][1] / rf) + voffset)
        upm_back = (int((coco_model.points[1][0] + mpi_model.points[14][0]) / 2 / rf) + hoffset, int((coco_model.points[1][1] + mpi_model.points[14][1]) / 2 / rf) + voffset)
        lom_back = (int(mpi_model.points[14][0] / rf) + hoffset, int(mpi_model.points[14][1] / rf) + voffset)
        lo_back = (int(mpi_model.points[14][0] / rf) + hoffset, int(1.05 * (mpi_model.points[14][1] + (coco_model.points[11][1] + coco_model.points[8][1]) / 2) / 2 / rf) + voffset)
        l_shoulder = (int(coco_model.points[5][0] / rf) + hoffset, int(coco_model.points[5][1] / rf) + voffset)
        r_shoulder = (int(coco_model.points[2][0] / rf) + hoffset, int(coco_model.points[2][1] / rf) + voffset)

        ldist = np.abs(coco_model.points[11][0] - mpi_model.points[14][0])
        rdist = np.abs(coco_model.points[8][0] - mpi_model.points[14][0])

        if ldist > rdist:
            lm_back = (int((mpi_model.points[14][0] - ldist) / rf) + hoffset, int(mpi_model.points[14][1] / rf) + voffset)
            rm_back = (int((mpi_model.points[14][0] + ldist) / rf) + hoffset, int(mpi_model.points[14][1] / rf) + voffset)
            llo_back = (int((mpi_model.points[14][0] - ldist) / rf) + hoffset, int(1.05 * (mpi_model.points[14][1] + (coco_model.points[11][1] + coco_model.points[8][1]) / 2) / 2 / rf) + voffset)
            rlo_back = (int((mpi_model.points[14][0] + ldist) / rf) + hoffset, int(1.05 * (mpi_model.points[14][1] + (coco_model.points[11][1] + coco_model.points[8][1]) / 2) / 2 / rf) + voffset)
        else:
            lm_back = (int((mpi_model.points[14][0] - rdist) / rf) + hoffset, int(mpi_model.points[14][1] / rf) + voffset)
            rm_back = (int((mpi_model.points[14][0] + rdist) / rf) + hoffset, int(mpi_model.points[14][1] / rf) + voffset)
            llo_back = (int((mpi_model.points[14][0] - rdist) / rf) + hoffset, int(1.05 * (mpi_model.points[14][1] + (coco_model.points[11][1] + coco_model.points[8][1]) / 2) / 2 / rf) + voffset)
            rlo_back = (int((mpi_model.points[14][0] + rdist) / rf) + hoffset, int(1.05 * (mpi_model.points[14][1] + (coco_model.points[11][1] + coco_model.points[8][1]) / 2) / 2 / rf) + voffset)
        
        self.polygons.append([neck, up_back, l_shoulder])
        self.polygons.append([neck, up_back, r_shoulder])
        self.polygons.append([up_back, upm_back, l_shoulder])
        self.polygons.append([up_back, upm_back, r_shoulder])
        self.polygons.append([l_shoulder, upm_back, lm_back])
        self.polygons.append([r_shoulder, upm_back, rm_back])
        self.polygons.append([upm_back, lom_back, lm_back])
        self.polygons.append([upm_back, lom_back, rm_back])
        self.polygons.append([lm_back, lom_back, lo_back, llo_back])
        self.polygons.append([rm_back, lom_back, lo_back, rlo_back])

    def print_stats(self, image):
        for i in range(len(self.polygons)):
            img = Image.new('L', (image.shape[1], image.shape[0]), 0)
            ImageDraw.Draw(img).polygon(self.polygons[i], outline=1, fill=1)
            mask = np.array(img, dtype=bool)

            avg = np.mean(image[mask])
            std = np.std(image[mask])
            min, pct25, pct50, pct75, max = np.percentile(image[mask], (0, 25, 50, 75, 100))

            print("Region:", i)
            print(f"Average:             {avg:.1f} ºC")
            print(f"Stdandard deviation:  {std:.1f} ºC")
            print(f"Min:                 {min:.1f} ºC")
            print(f"Max:                 {max:.1f} ºC")
            print(f"25th percentile:     {pct25:.1f} ºC")
            print(f"50th percentile:     {pct50:.1f} ºC")
            print(f"75th percentile:     {pct75:.1f} ºC")
            print()
            
    def display_histograms(self, image):
        values = []
        for i in range(len(self.polygons)):
            img = Image.new('L', (image.shape[1], image.shape[0]), 0)
            ImageDraw.Draw(img).polygon(self.polygons[i], outline=1, fill=1)
            mask = np.array(img, dtype=bool)
            values.append(image[mask])
                    
        fig, axs = plt.subplots(int(len(self.polygons)/2))
            
        for i in range(int(len(self.polygons)/2)):
            vals1 = values[2*i]
            vals2 = values[2*i+1]
                        
            minval, maxval = np.percentile(np.concatenate((vals1, vals2)), (0, 100))
            bins = np.linspace(minval, maxval, num=25)
            axs[i].hist(vals1, bins, alpha=0.5, label=f"Region {2*i}", edgecolor='black')
            axs[i].hist(vals2, bins, alpha=0.5, label=f"Region {2*i+1}", edgecolor='black')
            axs[i].legend(loc="upper left")
            
        plt.tight_layout()
        plt.show()

    def display(self, image):
        copy = np.copy(image)

        colors = np.zeros((10,1,3))
        colors[0,0,:] = [255, 0, 0]
        colors[1,0,:] = [0, 255, 0]
        colors[2,0,:] = [0, 0, 255]
        colors[3,0,:] = [255, 255, 0]
        colors[4,0,:] = [0, 255, 255]
        colors[5,0,:] = [255, 0, 255]
        colors[6,0,:] = [128, 0, 0]
        colors[7,0,:] = [0, 0, 128]
        colors[8,0,:] = [255, 255, 255]
        colors[9,0,:] = [0, 0, 0]

        for i in range(len(self.polygons)):
            img = Image.new('L', (copy.shape[1], copy.shape[0]), 0)
            ImageDraw.Draw(img).polygon(self.polygons[i], outline=1, fill=1)
            mask = np.array(img, dtype=bool)

            copy = np.multiply(copy, np.logical_not(np.resize(mask, (copy.shape[0], copy.shape[1], 1))))
            copy[:, :, 0] = copy[:, :, 0] + mask * np.squeeze(colors[i, :, 0])
            copy[:, :, 1] = copy[:, :, 1] + mask * np.squeeze(colors[i, :, 1])
            copy[:, :, 2] = copy[:, :, 2] + mask * np.squeeze(colors[i, :, 2])
            
            sumx = 0
            sumy = 0
            
            for p in self.polygons[i]:
                sumx += p[0]
                sumy += p[1]
            offset = 5
            meanx = int(sumx / len(self.polygons[i])) - offset
            meany = int(sumy / len(self.polygons[i])) + offset
            
            text = str(i)
            org = (meanx, meany)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = 255 - np.squeeze(colors[i, :, :])
            cv2.putText(copy, text, org, font, fontScale, color)
            
        return copy

class Model:

    def __init__(self, mode):
        proto_file = f"../models/{mode}/pose_deploy_linevec.prototxt"
        if mode == "coco":
            weights_file = f"../models/{mode}/pose_iter_440000.caffemodel"
            self.num_points = 18
            self.pose_pairs = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
        elif mode == "mpi": 
            weights_file = f"../models/{mode}/pose_iter_160000.caffemodel"
            self.num_points = 15
            self.pose_pairs = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
        self.net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    def detect_points(self, image, threshold=0.2):
        in_width = 368
        in_height = 368
        in_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(in_blob)
        maps = self.net.forward()

        # Empty list to store the detected keypoints
        self.points = []

        for i in range(self.num_points):
            # Confidence map corresponding to body's part
            prob_map = maps[0, i, :, :]

            # Finf global masima of the probmap
            _, prob, _, point = cv2.minMaxLoc(prob_map)

            # Scale the point to fit on the original image
            x = (image.shape[1] * point[0]) / maps.shape[3]
            y = (image.shape[0] * point[1]) / maps.shape[2]

            if prob > threshold : 
                # Add the point to the list if the probability is greater than the threshold
                self.points.append((int(x), int(y)))
            else:
                self.points.append(None)

    def display(self, image):
        copy = np.copy(image)
        for i in range(self.num_points):
            if self.points[i] is not None:
                cv2.circle(copy, (int(self.points[i][0]), int(self.points[i][1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(copy, "{}".format(i), (int(self.points[i][0]), int(self.points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        plt.figure()
        plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite("keypoints.png", copy)


        
