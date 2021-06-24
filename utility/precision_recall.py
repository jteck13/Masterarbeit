import numpy as np
import matplotlib.pyplot as plt


def print_printRecPrec(recallList, precisionList):
    #recallList = ([0.8571428571428571, 0.7142857142857143, 0.5714285714285714, 0.42857142857142855, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.2857142857142857, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #precisionList = ([0.07407407407407407, 0.0625, 0.05063291139240506, 0.038461538461538464, 0.025974025974025976, 0.02631578947368421, 0.02666666666666667, 0.02702702702702703, 0.0273972602739726, 0.027777777777777776, 0.028169014084507043, 0.02857142857142857, 0.028985507246376812, 0.029411764705882353, 0.029850746268656716, 0.030303030303030304, 0.03076923076923077, 0.015625, 0.015873015873015872, 0.016129032258064516, 0.01639344262295082, 0.016666666666666666, 0.01694915254237288, 0.017241379310344827, 0.017543859649122806, 0.017857142857142856, 0.01818181818181818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    recall=np.array(recallList)
    precision=np.array(precisionList)
    precision2=precision.copy()
    i=recall.shape[0]-2

    # interpolation...
    while i>=0:
        if precision[i+1]>precision[i]:
            precision[i]=precision[i+1]
        i=i-1

    # plotting...
    fig, ax = plt.subplots()
    for i in range(recall.shape[0]-1):
        ax.plot((recall[i],recall[i]),(precision[i],precision[i+1]),'k-',label='',color='red') #vertical
        ax.plot((recall[i],recall[i+1]),(precision[i+1],precision[i+1]),'k-',label='',color='red') #horizontal

    ax.plot(recall,precision2,'k--',color='blue')
    #ax.legend()
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    plt.savefig('fig.jpg')
    fig.show()