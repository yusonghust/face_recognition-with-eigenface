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
import matplotlib.pyplot as plt

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])#求每一个列向量的范数
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection
    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]

    #compute the within-class scatter matrix
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)

    #compute the between-class scatter matrix
    B = np.zeros((dim,dim))
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size

    #solve the generalized eigenvalue problem for discriminant directions
    ew, ev = linalg.eig(B, W)

    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
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
    [r,c] = A.shape#A是22400*120矩阵
    m = np.mean(A,1)#求每一行均值,m应该是1*22400的矩阵
    A = A - np.tile(m, (c,1)).T#零均值化,此时A任然是22400*120矩阵
    B = np.dot(A.T, A)
    [d,v] = linalg.eig(B)#求A.T*A的特征值d(返回一行向量,120个）与归一化的特征向量v,v的第i列对应第i个特征值

    # sort d in descending order
    order_index = np.argsort(d)
    order_index =  order_index[::-1]#将特征值从大到小排列
    d = d[order_index]
    v = v[:, order_index]#将特征向量按特征值排列

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)#根据课件 Av is eigenvector of AA.T,此时的W是AA.t的特征向量
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1#标准化特征矩阵？
    
    LL = d[0:-1]#特征值,省略最后一个

    W = W2[:,0:-1]  #omit last column, which is the nullspace,特征向量
    
    return W, LL, m

def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = [] # Label will store list of identity label
 
    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] =='bmp':
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

    return faces,idLabel

def float2uint8(arr):
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr-mmin)/(mmax-mmin)*255
    arr = np.uint8(arr)
    return arr


'''PCA feature'''
def PCA(k):
    dir='C:/Users/yu/Desktop/face/train'
    faces,idlable=read_faces(dir)
    [r,c]=np.shape(faces)
    W,LL,m=myPCA(faces)
    We=W[:,:k]
    y=np.dot(We.T,(faces-np.tile(m,(c,1)).T))
    x=np.dot(We,y)+np.tile(m,(c,1)).T
    return x,y,W,LL,m,We

'''LDA feature'''
def LDA(k):
    dir='C:/Users/yu/Desktop/face/train'
    faces,idlable=read_faces(dir)
    [r,c]=np.shape(faces)
    W,LL,m=myPCA(faces)
    W1=W[:,:k]
    x1=np.dot(W1.T,(faces-np.tile(m,(c,1)).T))
    Wf,C,classlabel=myLDA(x1,idlable)
    y=np.dot(Wf.T,np.dot(W1.T,(faces-np.tile(m,(c,1)).T)))
    return y,Wf,W1,C,classlabel
    
'''enrollment'''
def enrollment(y1,C):#y1 is for PCA,C is for LDA
    Z1=[]#PCA
    Z2=[]#LDA, Z2 is exactly the Centers returned by myLDA function
    for i in range(0,10):
        y11=y1[:,i*12:(i*12+12)]
        Z1.append(np.mean(y11,1))
    Z1=np.transpose(Z1)
    Z2=C
    return Z1,Z2
def distance(z,b):#计算欧氏距离
    x=z.shape[0]
    bb=np.tile(b,(x,1))
    dis=bb-z
    dis=dis**2
    Dis=dis.sum(axis=1)
    Dis=Dis**0.5
    
#    dis=linalg.norm(z-bb)
    return Dis
    
def ConMat(We,Wf,W1,m,z1,z2,alpha):
    CM1=np.zeros((10,10))
    CM2=np.zeros((10,10))
    dir='C:/Users/yu/Desktop/face/test'
    faces,idlabel=read_faces(dir)
    [r,c]=np.shape(faces)
    #将test的face投影到PCA,LDA空间中
    y3=np.dot(We.T,(faces-np.tile(m,(c,1)).T))#y3 is test for PCA
    y4=np.dot(Wf.T,np.dot(W1.T,(faces-np.tile(m,(c,1)).T)))#y4 is test  for LDA
    '''PCA'''
    success1=0
    for i in range(120):
        label=recognise(np.transpose(z1),np.transpose(y3[:,i]))
        CM1[idlabel[i],label]+=1
        if idlabel[i]==label:
            success1+=1
    rate_PCA=success1/120
    '''LDA'''
    success2=0
    for i in range(0,120):
        label=recognise(np.transpose(z2),np.transpose(y4[:,i]))
        CM2[idlabel[i],label]+=1
        if idlabel[i]==label:
            success2+=1
    rate_LDA=success2/120
       
    
    return rate_PCA,CM1,rate_LDA,CM2

def ConMat_Fus(We,Wf,W1,m,z1,z2,alpha):
    CM3=np.zeros((10,10))
    dir='C:/Users/yu/Desktop/face/test'
    faces,idlabel=read_faces(dir)
    [r,c]=np.shape(faces)
    #将test的face投影到PCA,LDA空间中
    y3=np.dot(We.T,(faces-np.tile(m,(c,1)).T))#y3 is test for PCA
    y4=np.dot(Wf.T,np.dot(W1.T,(faces-np.tile(m,(c,1)).T)))#y4 is test  for LDA
    y5=fusion(y3,y4,alpha)#y5 is for fusion
    z3=fusion(z1,z2,alpha)#z3 is for fusion
    '''fusion'''
    success3=0
    for i in range(0,120):
        label=recognise(np.transpose(z3),np.transpose(y5[:,i]))
        CM3[idlabel[i],label]+=1
        if idlabel[i]==label:
            success3+=1
    rate_Fus=success3/120 
    return rate_Fus,CM3
    
'''PCA recognise'''
def recognise(y1,y2):#y1 is for train,y2 is for test
    dis=distance(y1,y2)
    id=dis.argsort()[0]
    return id


def task2(m,We):
    M=float2uint8(m)
    M=np.reshape(M,(160,140))
    plt.figure("PCA",figsize=(12,12))
    plt.subplot(3,3,1)
    plt.title('mean')
    plt.imshow(M, cmap="gray")
    for i in range(0,8):
        eigface=We[:,i]
        eigface=float2uint8(eigface)
        eigface=np.reshape(eigface,(160,140))
        plt.subplot(3,3,i+2)
        plt.title('Eigenface'+str(i+1))
        plt.imshow(eigface, cmap="gray")
    plt.savefig('task2.jpg')
    plt.show( )
    
def task4(Cf,Wf,We,m):
    Cp=np.dot(Wf,Cf)
    Cr=np.dot(We,Cp)+np.tile(m,(10,1)).T
    plt.figure('LDA',figsize=(15,6))
    for i in range(10):
        center=Cr[:,i]
        center=float2uint8(center)
        center=np.reshape(center,(160,140))
        plt.subplot(2,5,i+1)
        plt.title('Center'+str(i+1))
        plt.imshow(center,cmap="gray")
    plt.savefig('task4.jpg')
    plt.show()
    
def fusion(ye,yf,alpha):#ye is for PCA,yf is for LDA
    y=np.vstack((alpha*ye,(1-alpha)*yf))
    return y
 
if __name__=='__main__':
    
    x,y,W,LL,m,We=PCA(30)
    b,Wf,W1,C,classlabel=LDA(90)
    Z1,Z2=enrollment(y,C)
    rate_PCA,CM1,rate_LDA,CM2=ConMat(We,Wf,W1,m,Z1,Z2,0.5)
    rate_Fus,CM3=ConMat_Fus(We,Wf,W1,m,Z1,Z2,0.5)
    '''q1'''
    print('The accuracy rate of PCA is',rate_PCA)
    print('The confusion matrix of PCA is','\n',CM1)
    print('The accuracy rate of LDA is',rate_LDA)
    print('The confusion matrix of LDA is','\n',CM2)
    print('The accuracy rate of Fusion is',rate_Fus)
    print('The confusion matrix of Fusion is','\n',CM3)
    task2(m,We)
    task4(C,Wf,W1,m)
    '''q3'''
    Rate=[]
    X=[]
    for i in range(1,10):
        alpha=0.1*i
        rate_Fus,CM3=ConMat_Fus(We,Wf,W1,m,Z1,Z2,alpha)
        Rate.append(rate_Fus)
        X.append(alpha)
    plt.figure('Accuracy of different alpha')
    plt.title('Accuracy of different alpha')
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')
    plt.plot(X,Rate)
    plt.savefig('fusion.jpg')
    plt.show()    
    
    
    

    
    