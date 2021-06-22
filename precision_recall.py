import numpy as np
import matplotlib.pyplot as plt

#openness
recall=np.array([0.071,.142,.214,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.285,.357,.428,.428])
precision=np.array([1,1,1,1,0.8,0.666,0.571,0.5,0.444,0.4,0.363,0.333,0.307,0.285,0.266,0.25,0.235,0.222,0.210,0.2,0.238,0.272,0.260])
#lrm
#recall=np.array([0.071,0.142,0.214,0.214,0.214,0.214,0.285,0.285,0.285,0.285])
#precision=np.array([1,1,1,0.75,0.6,0.5,0.571,0.5,0.5,0.5])

#lrm_exp
#recall=np.array([0.083,0.083,0.166,0.25,0.25,0.333,0.333,0.333])
#precision=np.array([1,0.5,0.666,0.75,0.6,0.666,0.571,0.5])
precision2=precision.copy()
i=recall.shape[0]-2

ap = 0.0

for x in precision:
    ap += ((recall[x]-recall[x-1])*precision[i])

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
ax.legend()
ax.set_xlabel("recall")
ax.set_ylabel("precision")
plt.xlim([0, 1])
#plt.savefig('fig.jpg')
fig.show()

APlrm = 1/11*(1*2+0.57)

APopen = 1/11*(3*1+.272)

APlrmExp = 1/11*(1+2*.75+0.66)
