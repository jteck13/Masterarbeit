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

path = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Img_ohne_Eval' + '\\'
annot = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Annotations\old_lrm' + '\\'

cpt = sum([len(files) for r, d, files in os.walk(path)])
print(cpt)

for e, i in enumerate(os.listdir(annot)):
    if e < 10:
        filename = i.split(".")[0] + ".png"
        print(filename)
        img = cv2.imread(os.path.join(path, '118.png'))
        df = pd.read_csv(os.path.join(annot, '118.csv'))
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

im = cv2.imread(os.path.join(path, "118.png"))
ss.setBaseImage(im)
ss.switchToSelectiveSearchQuality()
rects = ss.process()
imOut = im.copy()
for i, rect in (enumerate(rects)):
    x, y, w, h = rect
    #     print(x,y,w,h)
    #     imOut = imOut[x:x+w,y:y+h]
    cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
# plt.figure()
plt.imshow(imOut)
plt.show()


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


ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


train_images = []
train_labels = []

cntPosCum = 0
cntNegCum = 0

for e, i in enumerate(os.listdir(annot)):
    try:
        filename = i.split(".")[0] + ".png"
        print(e, filename)
        image = cv2.imread(os.path.join(path, filename))
        df = pd.read_csv(os.path.join(annot, i))
        gtvalues = []
        for row in df.iterrows():
            x1 = int(row[1][0].split(" ")[0])
            y1 = int(row[1][0].split(" ")[1])
            x2 = int(row[1][0].split(" ")[2])
            y2 = int(row[1][0].split(" ")[3])
            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchQuality()
        ssresults = ss.process()
        imout = image.copy()
        counter = 0
        falsecounter = 0
        flag = 0
        fflag = 0
        bflag = 0

        for e, result in enumerate(ssresults):
            if e < 4000 and flag == 0:
                for gtval in gtvalues:
                    x, y, w, h = result
                    iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    if counter < 30:
                        if iou > 0.5:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(1)
                            counter += 1
                    else:
                        fflag = 1
                    if falsecounter < 30:
                        if iou < 0.3:
                            if (cntPosCum >= cntNegCum):
                                timage = imout[y:y + h, x:x + w]
                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                            # print("outside")
                    else:
                        bflag = 1
                if fflag == 1 and bflag == 1:
                    print("inside")
                    flag = 1
        print(counter)
        cntPosCum = cntPosCum + counter
        cntNegCum = cntNegCum + falsecounter
    except Exception as e:
        print(e)
        print("error in " + filename)
        continue

print("Count positive samples: " +str(cntPosCum))
print("Count negative samples: " +str(cntNegCum))


X_new = np.array(train_images)
y_new = np.array(train_labels)

X_new.shape
y_new.shape

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle

print('#################################### Training ##############')
print('#################################### Training ##############')
print('#################################### Training ##############')
print('#################################### Training ##############')

BATCH_SIZE = 8
NAME = "resnet_openness_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
resnetModel = ResNet50(weights='imagenet', include_top=True)
resnetModel.summary()

for layers in (resnetModel.layers)[:15]:
    print(layers)
    layers.trainable = False

X = resnetModel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(resnetModel.input, predictions)
opt = Adam(lr=0.0001)
model_final.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
model_final.summary()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


lenc = MyLabelBinarizer()
Y = lenc.fit_transform(y_new)
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)
steps_per_epoch = len(X_train) // BATCH_SIZE
validation_steps = len(X_test) // BATCH_SIZE
print(steps_per_epoch, validation_steps)
trdata = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
traindata = trdata.flow(x=X_train, y=y_train, batch_size=BATCH_SIZE)
tsdata = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
testdata = tsdata.flow(x=X_test, y=y_test, batch_size=BATCH_SIZE)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("ieeercnn_resnet_lrm_1.h5", monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
history = model_final.fit_generator(generator=traindata, steps_per_epoch=steps_per_epoch, epochs=1000,
                                 validation_data=testdata, validation_steps=validation_steps,
                                 callbacks=[checkpoint, early, tensorboard])


############################ Evaluate ############################################################

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



############################ Load Model ##########################################################
model_saved = load_model('ieeercnn_resnet_lrm.h5')


########################### evaluate #############################################################


from sklearn.metrics import confusion_matrix

y_pred=model_final.predict(X_test, batch_size=8, verbose=1)
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

yhat = model_final.predict(X_test, batch_size= 8, verbose=1)

# reduce to 1d array
yhat_probs = yhat[:, 1]
yhat_classes = y_pred

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)



############################ Predict model #######################################################

pathOst = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Training_Ost_PNG' + '\\'
pathPred = r'C:\Users\jteck\Documents\Uni\Masterarbeit\Training\Images\28042021' + '\\'
z = 0

for e, i in enumerate(os.listdir(pathPred)):
    filenameRes = i.split(".")[0]
    z += 1
    img = cv2.imread(os.path.join(pathPred, i))
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
            out = model_final.predict(img)
            #out= model_saved.predict(img)
            #print(out[0][0])
            if out[0][0] > 0.9:
                #print(out[0][0])
                cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(imout, str("%.2f" % round(out[0][0],2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (36, 255, 12), 2)
    plt.figure()
    plt.imshow(imout)
    plt.show()
    cv2.imwrite('result/lrm/25052021_/result{}.png'.format(filenameRes), imout)
    # cv2.rectangle(copy,(s,u),(t,v),(0,0,255),2)

