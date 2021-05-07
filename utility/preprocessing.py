import os, cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


path = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Images\28042021' + '\\'
annot = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Annotations' + '\\'


cpt = sum([len(files) for r, d, files in os.walk(path)])
print(cpt)

bboxNew = []

img = cv2.imread(os.path.join(path, '118.png'))

for e, i in enumerate(os.listdir(annot)):
    if e < 10:
        filename = i.split(".")[0] + ".png"
        print(filename)
        img = cv2.imread(os.path.join(path, '118.png'))
        df = pd.read_csv(os.path.join(annot, '118.csv'))
        plt.imshow(img)
        plt.show()
        for row in df.iterrows():
            x1 = int(row[1][0].split(" ")[0])
            y1 = int(row[1][0].split(" ")[1])
            x2 = int(row[1][0].split(" ")[2])
            y2 = int(row[1][0].split(" ")[3])
            x11 = x1 + 3
            x21 = x2 - 3
            y11 = y1 + 3
            y21 = y2 - 3
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.rectangle(img, (x11, y11), (x21, y21), (0, 255, 0), 2)
            bbox = {'x1': x11,
                    'x2': x21,
                    'y1': y11,
                    'y2': y21}
            print(row)

            # add new bboxes to list
            bboxNew.append(bbox)
        plt.figure()
        plt.imshow(img)
        plt.show()
        break

print(len(bboxNew))
print(bboxNew[5])

print(bboxNew[5]['x1'])

s = bboxNew[5]['x1']
t = bboxNew[5]['x2']
u = bboxNew[5]['y1']
v = bboxNew[5]['y2']

start = {s, u}
end = {t, v}

ROI_number = 0
copy = img.copy()

print(s, t, u, v)

for c in bboxNew[5]:
    roi = img[u:v, s:t]
    cv2.imwrite('ROI/ROI_{}.png'.format(ROI_number), roi)
    cv2.rectangle(copy,(s,u),(t,v),(0,0,255),2)
    ROI_number += 1
    plt.figure()
    plt.imshow(copy)
    plt.show()
    break



############# insert rois in image #####################
copyNew = img.copy()
hole = cv2.imread('ROI_0.png')
plt.figure()
plt.imshow(hole)
plt.show()


x_offset = 30
y_offset = 170
x_end = x_offset + hole.shape[1]
y_end = y_offset + hole.shape[0]
copyNew[y_offset:y_end, x_offset:x_end] = hole
plt.figure()
plt.imshow(copyNew)
plt.show()
