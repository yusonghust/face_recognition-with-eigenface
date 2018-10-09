# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:35:12 2017

@author: yu
"""

import os
import numpy as np
import scipy.linalg as linalg
import cv2
import operator
import matplotlib
from matplotlib import pyplot as plt


def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1, col))

    for i in range(col):
        r[0, i] = linalg.norm(x[:, i])
    return r


def myLDA(A, Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection


    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim, datanum = A.shape
    totalMean = np.mean(A, 1)
    partition = [np.where(Labels == label)[0] for label in classLabels]
    classMean = [(np.mean(A[:, idx], 1), len(idx)) for idx in partition]

    # compute the within-class scatter matrix
    W = np.zeros((dim, dim))
    for idx in partition:
        W += np.cov(A[:, idx], rowvar=1) * len(idx)

    # compute the between-class scatter matrix
    B = np.zeros((dim, dim))
    for mu, class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset, offset) * class_size

    # solve the generalized eigenvalue problem for discriminant directions
    ew, ev = linalg.eig(B, W)
    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind, val in sorted_pairs[:classNum - 1]]
    LDAW = ev[:, selected_ind]
    Centers = [np.dot(mu, LDAW) for mu, class_size in classMean]
    Centers = np.array(Centers).T
    return LDAW, Centers, classLabels


def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r, c] = A.shape
    m = np.mean(A, 1)
    A = A - np.tile(m, (c, 1)).T
    B = np.dot(A.T, A)
    [d, v] = linalg.eig(B)

    # sort d in descending order
    order_index = np.argsort(d)
    order_index = order_index[::-1]
    print(order_index)
    d = d[order_index]
    v = v[:, order_index]

    # compute eigenvectors of scatter matrix
    W = np.dot(A, v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1

    LL = d[0:-1]

    W = W2[:, 0:-1]  # omit last column, which is the nullspace

    return W, LL, m


def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = []  # Label will store list of identity label

    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] == 'bmp':
            continue
        infile = os.path.join(directory, f)
        im = cv2.imread(infile, 0)
        # turn an array into vector
        im_vec = np.reshape(im, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A, dtype=np.float32)
    faces = faces.T
    idLabel = np.array(Label)

    return faces, idLabel


def float2uint8(arr):
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr - mmin) / (mmax - mmin) * 255
    arr = np.uint8(arr)
    return arr




def fusion(a,ye,yf):
    y=np.vstack((a*ye,yf*(1-a)))
    return y

def edistance(a,b):  #Calculate the Euclidean distance
    multi=b.shape[0]
    multia=np.tile(a,(multi,1))
    distance=multia-b
    distance=distance**2
    dissum=distance.sum(axis=1)
    dissum=dissum**0.5
    return dissum

def recognize(test,train):
    dist=edistance(test,train)
    lbl=dist.argsort()[0]
    return lbl

def cfmatrix(z,a,We, m, LDAW, W1,testdir,mode):      #Calculate the Confusion Matrix
    cm=np.zeros((10,10))
    faces,labels=read_faces(testdir)
    [r, c] = faces.shape
    y = np.dot(np.transpose(We), (faces - np.transpose(np.tile(m, (c, 1)))))
    y2 = np.dot(np.transpose(LDAW), np.dot(np.transpose(W1), faces - np.transpose(np.tile(m, (c, 1)))))
    if(mode==0):    #fusion
        zm=fusion(a,y,y2)
    if (mode == 1):   #PCA
        zm=y
    if (mode == 2):   #LDA
        zm=y2
    success=0
    for i in range(120):
        lbl=recognize(np.transpose(zm[:,i]),np.transpose(z))
        cm[labels[i],lbl]+=1
        if labels[i]==lbl:
            success+=1
    rate=success/120
    return cm,rate

def calFAR(m):    #Calculate the FAR
    farsum=0
    for i in range(m.shape[0]):
        FP=np.sum(m[:,i])-m[i][i]
        TN=np.sum(m)-np.sum(m[:,i])-np.sum(m[i,:])+m[i][i]
        far=FP/(FP+TN)
        farsum+=far
    FAR=farsum/m.shape[0]
    return FAR

def calFRR(m):    #Calculate the FRR
    frrsum=0
    for i in range(m.shape[0]):
        FN=np.sum(m[i,:])-m[i][i]
        TP=m[i][i]
        frr=FN/(FN+TP)
        frrsum+=frr
    FRR=frrsum/m.shape[0]
    return FRR






'''PCA feature'''
dir = "C:/Users/yu/Desktop/face/train"
faces, idlabel = read_faces(dir)
W, LL, m = myPCA(faces)
K = 30
[r, c] = faces.shape
We = W[:, : K]
y = np.dot(np.transpose(We), (faces - np.transpose(np.tile(m, (c, 1)))))
x = np.dot(We, y) + np.transpose(np.tile(m, (c, 1)))

''''LDA feature'''
K1 = 90
W1 = W[:, : K1]
x2 = np.dot(np.transpose(W1), (faces - np.transpose(np.tile(m, (c, 1)))))
LDAW, Centers, classLabels = myLDA(x2, idlabel)
y2 = np.dot(np.transpose(LDAW), np.dot(np.transpose(W1), faces - np.transpose(np.tile(m, (c, 1)))))

z = []
z2=[]
for i  in range(0, 10):
    y11 = y[:, (i * 12):(i * 12 + 12)]
    z.append(np.mean(y11, 1))
z = np.transpose(z)
for i in range(0, 10):
    y12 = y2[:, (i * 12):(i * 12 + 12)]
    z2.append(np.mean(y12, 1))
z2 = np.transpose(z2)
ylabel = []

'''Task 2'''
mpic = float2uint8(m)
mpic = np.reshape(mpic, (160, 140))
plt.subplot(3, 3, 1), plt.title('mean')
plt.imshow(mpic, cmap="gray"), plt.axis('off')

for i in range(8):
    eigen = We[:, i]
    eigenpic = float2uint8(eigen)
    eigenpic = np.reshape(eigenpic, (160, 140))
    title = "Eigenface " + str(i + 1)
    plt.subplot(3, 3, i + 2), plt.title(title)
    plt.imshow(eigenpic, cmap="gray"), plt.axis('off')

plt.savefig("task2(eigenfaces).ps")
plt.show()

'''Task 4'''
cp=np.dot(LDAW,Centers)
We2=W1 = W[:, : 90]
cr=np.dot(We2,cp)+np.transpose(np.tile(m, (10, 1)))
for i in range(10):
    center = cr[:, i]
    centerpic = float2uint8(center)
    centerpic = np.reshape(centerpic, (160, 140))
    title = "Center" + str(i + 1)
    plt.subplot(2,5, i + 1), plt.title(title)
    plt.imshow(centerpic, cmap="gray"), plt.axis('off')

plt.savefig("task4(centerfaces).ps")
plt.show()

'''Final Test'''
testdir = "C:/Users/yu/Desktop/face/test"

'''PCAtest'''
cm, rate = cfmatrix(z, 0, We, m, LDAW, W1, testdir, 1)
print('The confusion matrix of PCA is:',cm)
print('The accuracy of PCA is: ',rate)

'''LDAtest'''
cm, rate = cfmatrix(Centers, 0, We, m, LDAW, W1, testdir, 2)
print('The confusion matrix of LDA is:',cm)
print('The accuracy of LDA is: ',rate)

'''mixedtest'''
z11=fusion(0.5,z,Centers)
cm, rate = cfmatrix(z11, 0.5, We, m, LDAW, W1, testdir, 0)
print('The confusion matrix of mixed identifier is:',cm)
print('The accuracy of mixed identifier is: ',rate)

xx=[]
yy=[]
for i in range(11):
    z11=fusion(i/10,z,Centers)
    cm,rate=cfmatrix(z11,i/10,We,m,LDAW,W1,testdir,0)
    xx.append(i/10)
    yy.append(rate)
plt.figure()
plt.title("Accuracy of Fusion")
plt.xlabel('α')
plt.ylabel('Accuracy')
plt.plot(xx,yy)
plt.savefig("fused feature performance.ps")
plt.savefig("fused feature performance.png")
plt.show()
