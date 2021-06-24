
import os, cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50



cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

############################ Load Model ##########################################################


############################ Predict model #######################################################

# calculate iou from test and pred
def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    #print(iou)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# import the necessary packages
import numpy as np
#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    #print(boxes)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    iou = boxes[:,4]
    pred = boxes[:,5]
    #iou = boxes[:,4]
    #pred = boxes[:, 5]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = iou
    idxs = np.argsort(idxs)

# keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick]


#pathOpen = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Images\28042021' + '\\'
#annotOpen = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Annotations\old_lrm' + '\\'

pathOpen = 'test/openness/imgs'
annotOpen = 'test/openness/annot'
model_saved = load_model('ieeercnn_resnet_openness_final_res1.h5')

acc134 = []
iou_acc134 = []
acc73 = []
iou_acc73 = []


z = 0
cnt = 0
ct = 0
resultDict = []

for e, i in enumerate(os.listdir(pathOpen)):
    if i.startswith("134.png"):
        filenameRes = i.split(".")[0]
        z += 1
        #read test bbox
        df = pd.read_csv(os.path.join(annotOpen, '134.csv'))
        gtvalues = []
        predvalues = []
        for row in df.iterrows():
            x1 = int(row[1][0].split(" ")[0])
            y1 = int(row[1][0].split(" ")[1])
            x2 = int(row[1][0].split(" ")[2])
            y2 = int(row[1][0].split(" ")[3])
            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        img = cv2.imread(os.path.join(pathOpen, i))
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchQuality()
        ssresults = ss.process()
        imout = img.copy()
        for e, result in enumerate(ssresults):
            if e < 2000:
                x, y, w, h = result
                timage = imout[y:y + h, x:x + w]
                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out = model_saved.predict(img)
                if out[0][0] > 0.01:
                    for gtval in gtvalues:
                        #calculate iou for predicted img
                        iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                        thistuple = (x,y,x+w,y+h,iou,out[0][0])
                        resultDict.append(thistuple)
                        boundingBoxes = np.array(resultDict)

        # hier gehts weiter
        try:
            boundingBoxes
        except NameError:
            print("no boundingoxes found")
        else:
            iouEnd = []
            pick = non_max_suppression_slow(boundingBoxes, 0.1)
            for (startX, startY, endX, endY, iou, pred) in pick:
                color = list(np.random.random(size=3) * 256)
                #print(color)
                cv2.rectangle(imout, (int(startX), int(startY)), (int(endX), int(endY)), (color), 2)
                cv2.putText(imout, str("%.2f" % round(pred, 2)), (int(startX) - 0, int(startY) - 5 ),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (36, 255, 12), 2)
                if(iou > 0.0):
                    cv2.putText(imout, str("%.2f" % round(iou, 2)), (int(startX) + 15, int(startY) - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                .5, (255, 36, 12), 2)

                #iou_acc73.append(round(iou, 2))
                #acc73.append(round(pred, 2))

                iou_acc134.append(round(iou, 2))
                acc134.append(round(pred, 2))

                #print("iou")
                #print(round(iou, 2))
                #print("acc")
                #print(round(pred, 2))
        plt.figure()
        plt.imshow(imout)
        plt.show()
        # cv2.imwrite('result/openness/result{}.png'.format(filenameRes), imout)
        break

del boundingBoxes


#acc134 = [.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.05, 0.06, 0.06, 0.1, 0.12, 0.14, 0.16, 0.17, 0.22, 0.22, 0.28, 0.37, 0.43, 0.45, 0.6, 0.72, 0.75, 0.84, 0.91, 0.91, 0.92, 0.93, 0.94, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0]
#iou_acc134 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.93]

acc_all = acc134 + acc73
iou_all = iou_acc134 + iou_acc73

acc, iou_acc = zip(*sorted(zip(acc_all, iou_all)))

print(acc)
print(iou_acc)

prediction = []

for i,a in enumerate(acc):
    if(acc[i] > .5 and iou_acc[i] > 0.5):
        prediction.append("TP")
    else:
        prediction.append("FP")


print(prediction)

def precisionRecall(predi):
    index = 0
    acu_tp = 0
    acu_fp = 0
    prec_list= []
    recall_list = []
    for item in predi[::-1]:
        precision = 0
        recall = 0
        if predi[index] == 'TP':
            acu_tp = acu_tp + 1
        else:
            acu_fp = acu_fp + 1
        #print(acu_tp)
        precision = acu_tp/(acu_tp+acu_fp)
        recall = acu_tp/7
        prec_list.append(precision)
        recall_list.append(recall)
        index = index + 1
    return prec_list, recall_list




prec_list, recall_list = precisionRecall(prediction)