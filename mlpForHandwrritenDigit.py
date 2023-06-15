import os,math,csv,time,sys
import pandas as pd
import numpy as np
from numpy import genfromtxt
from collections import defaultdict
# https://github.com/Fodark/mlp-python
class MLP:
    def __init__(self, batchSize, learnRate, nEpochs):
        self.nEpochs = nEpochs
        self.batchSize = batchSize
        self.learnRate = learnRate
        
        # intialize the weights W and b for the hidden layers
        # Hidden Layer 1
        self.W_HL1=np.random.randn(300, 784) / np.sqrt(300)
        self.b_HL1=np.random.randn(300,1) / np.sqrt(300)
        # Hidden Layer 2
        self.W_HL2=np.random.randn(200, 300) / np.sqrt(200)
        self.b_HL2=np.random.randn(200,1) / np.sqrt(200)
        # Output Layer
        self.W_OL=np.random.randn(10, 200) / np.sqrt(10)
        self.b_OL=np.random.randn(10,1) / np.sqrt(10)
    
    # Divide Data into batches
    def getBatches(self, X):
        nData = X.shape[1] # n=data size
        batches = []
        nbatches = math.floor(nData/self.batchSize)
        for i in range(nbatches):
            bX = X[:, i * self.batchSize : (i+1) * self.batchSize]
            batches.append(bX)

        if nData % self.batchSize != 0:
            bX = X[:, self.batchSize * math.floor(nData / self.batchSize) : nData]
            batches.append(bX)

        return batches

    def forwardFeed(self, X):
        # Hidden Layer 1
        self.Y_HL1 = np.dot(self.W_HL1, X) + self.b_HL1
        self.A_HL1 = self.sigmoid(self.Y_HL1)
        # Hidden Layer 2
        self.Y_HL2 = np.dot(self.W_HL2, self.A_HL1) + self.b_HL2
        self.A_HL2 = self.sigmoid(self.Y_HL2)
        # Output Layer
        self.Y_OL=np.dot(self.W_OL, self.A_HL2) + self.b_OL
        self.A_OL=self.softmax(self.Y_OL)
        #print('self.A_OL=',self.A_OL)
        # self.Y_OL=np.dot(self.W_OL, self.A_HL1) + self.b_OL
        # self.A_OL=self.softmax(self.Y_OL)
    
    # Activation function: Sigmoid
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    # Sigmoid Backward
    # https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
    def sigmoidBackward(self, dA, Y):
        sig = self.sigmoid(Y)
        return dA * sig * (1 - sig)
    
    # Activation function: Softmax
    def softmax(self, x):
        exps = np.exp(x - x.max())
        #print('x.max(=)',x.max())
        #print('np.sum(exps, axis=0)=',np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
 
    def train(self, X, Y):
        nData=X.shape[1]
        accArr=[]

        # Epochs loop
        for i in range(0, self.nEpochs):
            # shuffle data
            random = np.arange(nData)
            np.random.shuffle(random)
            xShuffle = X[:,random]
            yShuffle = Y[:,random]
            
            # divide data into batches
            xBatches=self.getBatches(xShuffle)
            yBatches=self.getBatches(yShuffle)
            # Batch loop
            for idx,(xBatch,yBatch) in enumerate(zip(xBatches,yBatches)):
                #print(idx,'th batch')
                self.forwardFeed(xBatch)

                # Backward Propgation
                dY_OL = self.A_OL - yBatch
                    # Output Layer Gradients
                self.dW_OL=np.dot(dY_OL,self.A_HL2.T)/self.batchSize
                #self.db_OL=np.sum(dY_OL, axis=1) / self.batchSize
                self.db_OL=np.sum(dY_OL, axis=1, keepdims=True) / self.batchSize
                    # Back propagation to Hidden layer 2
                dA_HL2 = np.dot(self.W_OL.T, dY_OL)
                dY_HL2 = self.sigmoidBackward(dA_HL2,self.Y_HL2)
                    # Hidden layer 2 Gradients
                self.dW_HL2 = np.dot(dY_HL2, self.A_HL1.T) / self.batchSize
                self.db_HL2 = np.sum(dY_HL2, axis=1, keepdims=True) / self.batchSize
                    # Back propagation to Hidden layer 1
                dA_HL1 = np.dot(self.W_HL2.T, dY_HL2)
                dY_HL1 = self.sigmoidBackward(dA_HL1,self.Y_HL1)
                    # Hidden layer 2 Gradients
                self.dW_HL1 = np.dot(dY_HL1, xBatch.T) / self.batchSize
                self.db_HL1 = np.sum(dY_HL1, axis=1, keepdims=True) / self.batchSize

                # Update parameters
                self.W_OL-=self.learnRate*self.dW_OL
                self.b_OL-=self.learnRate*self.db_OL
                self.W_HL2-=self.learnRate*self.dW_HL2
                self.b_HL2-=self.learnRate*self.db_HL2
                self.W_HL1-=self.learnRate*self.dW_HL1
                self.b_HL1-=self.learnRate*self.db_HL1

            # calculate train accuracy after each epoch
            trainAccuracy = self.getAccuracy(X, Y)
            accArr.append([i,trainAccuracy])
            print('Epoch',i,'trainAccuracy:',trainAccuracy)
            #if(trainAccuracy>0.99): break
        return accArr

    def getAccuracy(self, X, y):
        predictions = []

        self.forwardFeed(X)
        output = self.A_OL
        pred = np.argmax(output, axis=0)
        predictions.append(pred == np.argmax(y, axis=0))
        
        return np.mean(predictions)
            

if __name__=='__main__':
    # python NeuralNetwork3.py train_image.csv train_label.csv test_image.csv
    # fTrainImage=sys.argv[1]
    # fTrainLabel=sys.argv[2]
    # fTestImage=sys.argv[3]
    fTrainImage='data/train_image.csv'
    fTrainLabel='data/train_label.csv'
    fTestImage='data/test_image.csv'
    
    # with open('test_image.csv') as csv_file:
    #     csv_reader=csv.reader(csv_file,delimiter=',')
    #     for i,row in enumerate(csv_reader):
    #         testImg.append(row)
            
    # testImg=np.array(testImg).astype(int)
    # print('shap=',testImg.shape)

    #  Load Training and Testing data
    # https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy

    trainImage=pd.read_csv(fTrainImage, sep=',',header=None).to_numpy().T
    trainLabel=pd.read_csv(fTrainLabel, sep=',',header=None).to_numpy().T
    testImage=pd.read_csv(fTestImage, sep=',',header=None).to_numpy().T
    
    print('trainImg.shape=',trainImage.shape)
    print('trainLbl.shape=',trainLabel.shape)
    print('testImg.shape=',testImage.shape)
    
    # Organize Input X / Output Y format
    xTrain=trainImage
    yTrain = np.zeros((10,trainLabel.size)) # 10: 0-9 number
    yTrain[trainLabel, np.arange(trainLabel.size)] = 1
    
    # Trim data size
    nData=60000
    xTrain = xTrain[:,0:nData]
    yTrain = yTrain[:,0:nData]

    # Training
    mlp = MLP(batchSize=100, learnRate=0.01, nEpochs=50) 
    startTime = time.time()
    accArr = mlp.train(xTrain, yTrain) # Epoch 30 Accuracy: 98.31%
    endTime = time.time()
    CPUtime=endTime-startTime
    print('Training CPU Time(s)=', CPUtime) 

    # Validation & Load Test Data
    fTestLabel='data/test_label.csv'
    testLabel=pd.read_csv(fTestLabel, sep=',',header=None).to_numpy().T
    print('testLbl.shape=',testLabel.shape)
    yTest = np.zeros((10,testLabel.size)) # 10: 0-9 number
    yTest[testLabel, np.arange(testLabel.size)] = 1
    testAccuracy = mlp.getAccuracy(testImage, yTest)
    print("Test Accuray=",testAccuracy) # 0.9206

    # Prediction
    mlp.forwardFeed(testImage)
    predDistribution = mlp.A_OL
    prediction = np.argmax(predDistribution, axis=0)
    df=pd.DataFrame(prediction)
    # print('prediction=',prediction)
    # print('df=',df)
    # https://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.DataFrame.to_csv.html
    df.to_csv('test_predictions.csv', header=False, index=False)


    
