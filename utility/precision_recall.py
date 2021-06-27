from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def print_printRecPrec(recallList, precisionList, label, thresh):

    count = 0
    n = 0.1
    apAry = []

    while count < len(recallList):
        # np.interp(n, recallList, precisionList)
        apAry.append(np.interp(n, recallList, precisionList))
        count = count + 1
        n = n + 0.1

    #print("ap_mean2")
    #print(1 / 11 * (sum(apAry)))

    recall = np.array(recallList)
    precision = np.array(precisionList)
    i = np.array(range(1, len(recall)))
    re = recall[i] - recall[i - 1]
    pr = precision[i] + precision[i - 1]

    # Multiply re and pr lists and then take sum and divide by 2
    ap = np.sum(re * pr) / 2
    print(round(ap, 2))

    precision2 = precision.copy()
    i = recall.shape[0] - 2

    # interpolation...
    while i >= 0:
        if precision[i + 1] > precision[i]:
            precision[i] = precision[i + 1]
        i = i - 1

    # plotting...
    fig, ax = plt.subplots()
    for i in range(recall.shape[0] - 1):
        ax.plot((recall[i], recall[i]), (precision[i], precision[i + 1]), 'k-', label='', color='red')  # vertical
        ax.plot((recall[i], recall[i + 1]), (precision[i + 1], precision[i + 1]), 'k-', label='',
                color='red')  # horizontal

    ax.plot(recall, precision2, 'k--', color='blue')
    # ax.legend()
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    fig.suptitle(label + str(thresh), fontsize=12)
    plt.xticks([0.0, 0.1, 0.2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    #plt.savefig('fig.jpg')
    fig.show()





