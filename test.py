import os
import numpy as np
import scipy.linalg as linalg
import cv2
import operator

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection


    classLabels = np.unique(Labels)#Find the unique elements of an array.
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)#average of rows
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
    ew, ev = linalg.eig(B,W)
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
    [r,c] = A.shape #Get rows,cols of A
    m = np.mean(A,1)#Get mean of columns
    A = A - np.tile(m, (c,1)).T
    #numpy.tile:Construct an array by repeating A the number of times given by reps.
    B = np.dot(A.T, A)
    [d,v] = linalg.eig(B)

    # sort d in descending order
    order_index = np.argsort(d)
    order_index =  order_index[::-1]#reverse
    print(order_index)
    d = d[order_index]
    v = v[:, order_index]

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1
    
    LL = d[0:-1]

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
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
    a,b=faces.shape
    print(a,b)
    idLabel = np.array(Label)
    print(idLabel)
    print(len(idLabel))

    return faces,idLabel

def float2uint8(arr):
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr-mmin)/(mmax-mmin)*255
    arr = np.uint8(arr)
    return arr

def PCA_feature():
    dir='C:/Users/yu/Desktop/face/train'
    faces,idlable=read_faces(dir)
    W,LL,me=myPCA(faces)
    We=W[:,:30]
    me=np.array(me).T
    We=We.T
    Y=[]
    [a,b]=faces.shape
    for x in range(0,b):
        x0=faces[:,x]-me
        y=np.dot(We,x0)
        Y.append(y)
    return Y,We

#    We=W[30,:]
#    tem= np.tile(me, (b,1)).T
#    faces = faces - tem
#    We=We.T
#    Y=[]
#    for x in range(0,b):
#        X=faces[:,x]-me
#        y=np.dot(We,X)
#        Y.append(y)
#    return Y
#    W=W.T
#    We=W[:,:30]
#    a,b=We.shape
    
def LDA_feature():
    dir='C:/Users/yu/Desktop/face/train'
    faces,idlable=read_faces(dir)
    W,LL,me=myPCA(faces)
    W1=W[:,:90]
    W1=W1.T
    [a,b]=faces.shape
    X=[]
    for x in range(0,b):
        x0=faces[:,x]-me
        x1=np.dot(W1,x0)
        X.append(x1)
    X=np.array(X, dtype=np.float32)
    X=X.T
    Wf,C,classLabels=myLDA(X,idlable)
    Wf=Wf.T
    temp=np.dot(Wf,W1)
    Y=[]
    for x in range(0,b):
        x0=faces[:,x]-me
        y=np.dot(temp,x0)
        Y.append(y)
    return Y,C
def enrollment():
    Y_PCA,W=PCA_feature()
    Y_LDA,Z_LDA=LDA_feature()
    Y_PCA=np.array(Y_PCA).T
    Y_LDA=np.array(Y_LDA).T
    Z_PCA=np.mean(Y_PCA,1)
    return Y_PCA,Y_LDA

def recognise():
    dir='C:/Users/yu/Desktop/face/test'
    faces,idlable=read_faces(dir)
    Y_PCA,W=PCA_feature()
    [x,y]=faces.shape
    b=len(Y_PCA)
    C=np.zeros((120,120))
    index=[]
    for x in range(0,y):
        distance=[]
        face=faces[:,x]
        q=np.dot(W,face)
        for i in range(0,b):
            temp=linalg.norm(Y_PCA[i]- q)
            distance.append(temp)
        minDistance = min(distance)
        index = distance.index(minDistance)
        print(index)
        
        
        
        
        
    

    
#Y=PCA_feature()
#LDA_feature()
#print(Y)
if __name__=='__main__':
   # Y_PCA,Y_LDA=enrollment()
    recognise()
    

        
    