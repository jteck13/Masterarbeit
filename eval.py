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



pathShow = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\vorübergehend\img' + '\\'
annotShow = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Annotations\old_lrm' + '\\'

cpt = sum([len(files) for r, d, files in os.walk(pathShow)])
print(cpt)

for e, i in enumerate(os.listdir(pathShow)):
    if e < 10:
        filename = i.split(".")[0] + ".png"
        print(filename)
        img = cv2.imread(os.path.join(pathShow, '119.png'))
        df = pd.read_csv(os.path.join(annotShow, '119.csv'))
        #plt.imshow(img)
        #plt.show()
        for row in df.iterrows():
            x1 = int(row[1][0].split(" ")[0])
            y1 = int(row[1][0].split(" ")[1])
            x2 = int(row[1][0].split(" ")[2])
            y2 = int(row[1][0].split(" ")[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.figure()
        plt.imshow(img)
        plt.show()
        break



cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

############################ Load Model ##########################################################
model_saved = load_model('ieeercnn_resnet_openness_1_exp1.h5')

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



pathOst = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Training_Ost_PNG' + '\\'
pathPred = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Images\28042021' + '\\'
resultPath = r'C:\Users\jteck\Documents\Uni\Masterarbeit\python\result\lrm'

pathOpen = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Images\12052021' + '\\'
annotOpen = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Annotations\new_120521' + '\\'


z = 0
tp = []
fp = []
fn = []
cnt = 0


for e, i in enumerate(os.listdir(pathOpen)):
    if i.startswith("184.png"):
        filenameRes = i.split(".")[0]
        z += 1
        #read test bbox
        df = pd.read_csv(os.path.join(annotOpen, '184.csv'))
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
            if e < 4000:
                x, y, w, h = result
                timage = imout[y:y + h, x:x + w]
                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out = model_saved.predict(img)
                iou_nomaxNew = {"x1": x, "x2": x + w, "y1": y, "y2": y + h}
                #out= model_saved.predict(img)
                #print(out[0][0])
                if out[0][0] > 0.0:
                    for gtval in gtvalues:
                        #print(x,y,x+w,y+h)
                        #calculate iou for predicted img
                        iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                        # print(out[0][0])
                        #cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                        #cv2.putText(imout, str("%.2f" % round(out[0][0],2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (36, 255, 12), 2)
                        if(iou > .5):
                            cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(imout, str("%.2f" % round(out[0][0], 2)), (x - 20, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (36, 255, 12), 2)
                            cv2.putText(imout, str("%.2f" % round(iou, 2)), (x - 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX,.5, (255, 36, 12), 2)
                            cnt += 1
                            print(iou)
                            print(out[0][0])
        plt.figure()
        plt.imshow(imout)
        plt.show()
        print(cnt)
        #cv2.imwrite('result/openness/result{}.png'.format(filenameRes), imout)
        break

