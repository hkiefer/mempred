import numpy as np
import pandas as pd
from numba import jit

@jit(nogil=True, nopython=True)
def jitCondCorr(xs,ys,xIds,cLen,tLen):
    if (len(xs) != len(ys)) or (len(xs) != len(xIds)):
        print(len(xs),len(ys),len(xIds))
        raise ValueError("Input array dimensions must match!")
    cc = np.zeros((cLen,tLen))
    ccc = np.zeros(cc.shape)
    for i in range(len(xs)-len(cc[0])):
        for j,y in enumerate(ys[i:i+len(cc[0])]):
            cc[xIds[i]][j]+=xs[i]*y
            ccc[xIds[i]][j]+=1
    return cc/ccc

def condCorrelation(a, conditionBinMidPoints, b=None, c=None, tLen = None, subtract_mean=False):
    meana = int(subtract_mean)*np.mean(a)
    a2 = a-meana
    if b is None:
        b2 = a2
    else:
        meanb = int(subtract_mean)*np.mean(b)
        b2 = b-meanb
    diffs = np.array(conditionBinMidPoints[1:]-conditionBinMidPoints[:-1])
    bins = conditionBinMidPoints[:-1]+diffs/2
    if c is None:
        c = a
    xIds = np.digitize(c,bins)
    if tLen is None:
        tLen = a.shape[0]
    return jitCondCorr(a2.tolist(),b2.tolist(),xIds.tolist(),conditionBinMidPoints.shape[0],tLen)

def pdCondCorr(df,f1,f2,conditionBinMidPoints, fc=None,trunc=None):

    a=df.loc[:,f1].values
    if not trunc is None:
        tLen=df[df.index<trunc].shape[0]
    else:
        tLen=a.shape[0]

    if fc is None:
        if f1 == f2:
            corr=condCorrelation(a,conditionBinMidPoints,tLen=tLen)
        else:
            b=df.loc[:,f2].values
            corr=condCorrelation(a,conditionBinMidPoints,b=b,tLen=tLen)
    else:
        c=df.loc[:,fc].values
        if f1 == f2:
            corr=condCorrelation(a,conditionBinMidPoints,c=c,tLen=tLen)
        else:
            b=df.loc[:,f2].values
            corr=condCorrelation(a,conditionBinMidPoints,b=b,c=c,tLen=tLen)

    ts = (df.index[:tLen]-df.index[0]).values
    cf=pd.DataFrame(corr.T, index=ts, columns=conditionBinMidPoints)
    return cf
