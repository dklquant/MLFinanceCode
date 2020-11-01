#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:44:29 2020

@author: jiajilu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from math import floor
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn import decomposition
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
from sklearn import datasets

class SwapMLData:
    
    def __init__(self, sd=None, ed=None):
        self.swapFly = None
        self.swap5Y = None
        self.swap10Y = None
        self.swap30Y = None
        self.spy = None
        self.igIndex = None
        self.libor1Y = None
        self.libor3M = None
        self.usdIndex = None
        self.joinData = None
        self.sd = sd
        self.ed = ed
        
    def loadData(self):
        self.swap5Y = pd.read_csv('DSWP5.csv', parse_dates=['DATE']).set_index('DATE')
        self.swap10Y = pd.read_csv('DSWP10.csv', parse_dates=['DATE']).set_index('DATE')
        self.swap30Y = pd.read_csv('DSWP30.csv', parse_dates=['DATE']).set_index('DATE')
        
        self.spy = pd.read_csv('SP500.csv', parse_dates=['DATE']).set_index('DATE')
        self.igIndex = pd.read_csv('IG.csv', parse_dates=['DATE']).set_index('DATE')
        
        self.libor1Y = pd.read_csv('USD12MD156N.csv', parse_dates=['DATE']).set_index('DATE')
        self.libor3M = pd.read_csv('USD3MTD156N.csv', parse_dates=['DATE']).set_index('DATE')
        
        self.usdIndex = pd.read_csv('DX-Y.NYB.csv', parse_dates=['Date']).set_index('Date')
        
        return
    
    def preProcess(self):
        self.swap5Y = self.swap5Y[self.sd:self.ed]
        self.swap10Y = self.swap10Y[self.sd:self.ed]
        self.swap30Y = self.swap30Y[self.sd:self.ed]
        
        self.spy = self.spy[self.sd:self.ed]
        self.igIndex = self.igIndex[self.sd:self.ed]
        
        self.libor3M = self.libor3M[self.sd:self.ed]
        self.libor1Y = self.libor1Y[self.sd:self.ed]

        self.usdIndex = self.usdIndex[self.sd:self.ed]['Adj Close']
        
        joinData = pd.concat([self.swap5Y, self.swap10Y, self.swap30Y, self.libor1Y,
                              self.spy, self.igIndex, self.libor3M, self.usdIndex],
                              axis=1
                   )
        self.joinData = joinData.apply(pd.to_numeric, errors='coerce')
        self.joinData = self.joinData.fillna(method='ffill')
        
        self.joinData.columns = ['Swap5Y', 'Swap10Y', 'Swap30Y', 'Libor1Y',
                                 'SPY', 'IG', 'Libor3M', 'USD']
        
        return
    
    def processData(self):
        self.preProcess()
        self.swap5Y = self.joinData.Swap5Y
        self.swap10Y = self.joinData.Swap10Y
        self.swap30Y = self.joinData.Swap30Y
        
        self.libor3M = self.joinData.Libor3M
        self.libor1Y = self.joinData.Libor1Y
        
        self.spy = self.joinData.SPY
        self.igIndex = self.joinData.IG
        self.usdIndex = self.joinData.USD
    
        self.swapFly = 2 * self.swap10Y - self.swap5Y - self.swap30Y
        
        return
    
class Features:
    
    def __init__(self, swapData):
        self.swapData = swapData
        self.zscore5d = None
        self.zscore1M = None
        self.zscore3M = None
        self.zscore6M = None
        self.zscore1Y = None
        
        self.level = None
        
        self.libor3M = None
        self.libor1Y = None
        self.swap5Y = None
        self.swap10Y = None
        self.swap30Y = None
        
        self.spy = None
        self.usdIndex = None
        self.igIndex = None
        self.swapFly = None
        
        self.labels = None
        
        self.featuresAndDep = None
        
    def computeZScore(self, var, d):
        m = var.rolling(window=d).mean()
        sd = var.rolling(window=d).std()        
        return (m / sd).rename('zscore{0}d'.format(d))
    
    def computeZScores(self):
        self.swapFly = self.swapData.swapFly
        self.zscore5d = self.computeZScore(self.swapFly, 5)
        self.zscore1M = self.computeZScore(self.swapFly, 22)
        self.zscore3M = self.computeZScore(self.swapFly, 66)
        self.zscore6M = self.computeZScore(self.swapFly, 132)
        self.zscore1Y = self.computeZScore(self.swapFly, 252)
        return
    
    def determineLabel(self, x):
        if np.isnan(x):
            return np.nan
        if x > 2:
            return 0
        elif x < -2:
            return 1
        else:
            return 2
        
    def generateLabels(self):
        std = self.swapFly.rolling(22).std()
        swapFlyForward = self.swapFly.shift(-7)
        swapFlyDiff = (swapFlyForward - self.swapFly) / std
        self.labels = swapFlyDiff.apply(self.determineLabel).rename('label')
        return
        
    def generateFeatures(self):
        self.computeZScores()
        self.level = self.swapFly
        ret = self.swapData.joinData / self.swapData.joinData.shift(1)
        self.ret = ret.apply(lambda x: np.log(x))
        
        self.libor3M = self.ret.Libor3M
        self.libor1Y = self.ret.Libor1Y
        self.spy = self.ret.SPY
        self.igIndex = self.ret.IG
        self.usdIndex = self.ret.USD
        self.swap5Y = self.ret.Swap5Y
        self.swap10Y = self.ret.Swap10Y
        self.swap30Y = self.ret.Swap30Y
        
        return
        
    def combineFeatures(self):
        self.generateFeatures()
        self.generateLabels()
        self.featuresAndDep = pd.concat([self.zscore5d, self.zscore1M, self.zscore3M,
                                         self.zscore6M, self.zscore1Y, self.level,
                                         self.libor3M, self.libor1Y, self.swap5Y,
                                         self.swap10Y, self.swap30Y, self.spy,
                                         self.igIndex, self.usdIndex, self.labels],
                                        axis=1
                              )
        self.featuresAndDep = self.featuresAndDep.replace([np.inf, -np.inf], np.nan)
        self.featuresAndDep = self.featuresAndDep.dropna()
        self.features = self.featuresAndDep.drop('label', axis=1)
        self.labels = self.featuresAndDep['label']
        return
        
class SwapMLFactory:
    
    def __init__(self):
        self.mlRegistry = {}
        self.registerML()
        
    def registerML(self):
        self.mlRegistry['DecisionTree'] = SwapMLDecisionTree
        self.mlRegistry['NeuralNet'] = SwapMLNeuralNetwork
        self.mlRegistry['GradientBoosting'] = SwapMLGradientBoosting
        self.mlRegistry['KNN'] = SwapMLKNN
        self.mlRegistry['SVM'] = SwapMLSVM

    def create(self, algo, features):
        return self.mlRegistry.get(algo)(features) 
    

class SwapMLBase:
    
    def __init__(self, features):
        self.features = features
        self.feature = self.features.features
        self.label = self.features.labels
        self.config = {}
        
    def formTrainingData(self, pct=0.7):
        pass
    
    def formTestingData(self, pct=0.3):
        pass
        
    def inSampleAccuracy(self):
        pass
    
    def outOfSampleAccuracy(self):
        pass
    
    def splitData(self, n=10):
        trainingDataSplit = []
        testingDataSplit = []
        labelTraining = []
        labelTesting = []
        totalSize = len(self.feature)
        delta = totalSize // 10
        for i in range(n - 1):
            trainingDataSplit.append(self.feature.iloc[:min(totalSize, 
                                                            (i + 1) * delta)]
                                    )
            testingDataSplit.append(self.feature.iloc[min(totalSize,
                                                         (i + 1) * delta):
                                                      min(totalSize,
                                                         (i + 2) * delta)]
                                   )
            labelTraining.append(self.label.iloc[:min(totalSize, 
                                                        (i + 1) * delta)]
                                )
            labelTesting.append(self.label.iloc[min(totalSize,
                                                    (i + 1) * delta):
                                                min(totalSize,
                                                    (i + 2) * delta)]
                                )
        return trainingDataSplit, testingDataSplit, labelTraining, labelTesting
        
    def rollingCrossValidation(self, n=10):
        trainingData, testingData, labelTraining, labelTesting = self.splitData(n)
        accuracy = []
        accuracy_train = []
        trainingLen = 1
        for train, test, labelTrain, labelTest in zip(trainingData, testingData, 
                                                      labelTraining, labelTesting):
            clf = self.createClf(train, labelTrain)
            pred = self.predict(clf, test)
            score = self.accuracyScore(pred, labelTest)
            accuracy.append([trainingLen, score])
            predTraining = self.predict(clf, train)
            score = self.accuracyScore(predTraining, labelTrain)
            accuracy_train.append([trainingLen, score])
            trainingLen += 1
        return accuracy, accuracy_train
    
    def nonRollingTrainTestError(self, n=10):
        size = len(self.feature)
        delta = size // n
        re_test, re_train = [], []
        for i in range(n - 1):
            idx = min((i + 1) * delta, size)
            train = self.feature.iloc[:idx]
            trainLabel = self.label.iloc[:idx]
            test = self.feature.iloc[idx:]
            testLabel = self.label[idx:]
            
            clf = self.createClf(train, trainLabel)
            pred = self.predict(clf, test)
            score = self.accuracyScore(pred, testLabel)
            re_test.append([i + 1, score])
            predTrain = self.predict(clf, train)
            score = self.accuracyScore(predTrain, trainLabel)
            re_train.append([i + 1, score])
            
        return re_train, re_test
    
    def nonRollingCrossValidation(self, n=5):
        clf = self.createClf(self.feature, self.label)
        score = cross_val_score(clf, self.feature, self.label, cv=n)
        return score
        
    def createClf(self, train, trainLabel):
        raise NotImplementedError('Have to be implemented')
    
    def predict(self, clf, test):
        return clf.predict(test)
    
    def accuracyScore(self, pred, label):
        return accuracy_score(label, pred)
    
    def cvAgg(self, isRolling=True):
        if isRolling:
            _, score = self.rollingCrossValidation()
        else:
            score = self.nonRollingCrossValidation()
        return np.array(score)[:, 1]
    
    def crossValidationScore(self, testScore):
        return float(np.mean(testScore))
    
        
class SwapMLDecisionTree(SwapMLBase):
    
    def __init__(self, features):
        super().__init__(features)
        self.config = {'alpha': 0.1}
        
    def createClf(self, train, trainLabel):
        return DecisionTreeClassifier(random_state=0, 
                                      ccp_alpha=self.config['alpha']).fit(train, 
                                                                          trainLabel
                                     )
    
    def crossValidation(self, rolling):
        re = []
        for alpha in [0.01, 0.04, 0.07, 0.1, 0.15, 0.2, 0.5]:
            self.config['alpha'] = alpha
            re.append([alpha, 
                       self.crossValidationScore(
                           self.cvAgg(isRolling=rolling)
                           )
                       ]
                      )
        return re
        
class SwapMLGradientBoosting(SwapMLBase):

    def __init__(self, features):
        super().__init__(features)
        self.config = {'depth': 0.1}

    def createClf(self, train, trainLabel):
        return GradientBoostingClassifier(random_state=0, 
                                          max_depth=self.config['depth']).fit(train, 
                                                                          trainLabel
                                         )

    def crossValidation(self, rolling):
        re = []
        for depth in [3, 4, 8, 13, 15, 17, 20, 50]:
            self.config['depth'] = depth
            re.append([depth, 
                       self.crossValidationScore(
                           self.cvAgg(isRolling=rolling)
                           )
                       ]
                      )
        return re
        
class SwapMLNeuralNetwork(SwapMLBase):
    
    def __init__(self, features):
        super().__init__(features)
        self.config = {'depth': 100}

    def createClf(self, train, trainLabel):
        return MLPClassifier(random_state=1, 
                             hidden_layer_sizes=(self.config['depth'],)).fit(train, 
                                                                             trainLabel
                            )
                                                                             
    def crossValidation(self, rolling):
        re = []
        for depth in [10, 20, 30, 50, 100, 150, 200]:
            self.config['depth'] = depth
            re.append([depth, 
                       self.crossValidationScore(
                           self.cvAgg(isRolling=rolling)
                           )
                       ]
                      )
        return re

class SwapMLKNN(SwapMLBase):

    def __init__(self, features):
        super().__init__(features)
        self.config = {'k': 3}

    def createClf(self, train, trainLabel):
        return KNeighborsClassifier(n_neighbors=self.config['k']).fit(train, trainLabel)

    def crossValidation(self, rolling):
        re = []
        for k in [1, 2, 3, 5, 10]:
            self.config['k'] = k
            re.append([k, 
                       self.crossValidationScore(
                           self.cvAgg(isRolling=rolling)
                           )
                       ]
                      )
        return re


class SwapMLSVM(SwapMLBase):

    def __init__(self, features):
        super().__init__(features)
        self.config = {'c': 3}

    def createClf(self, train, trainLabel, kernel='rbf'):
        return SVC(kernel=kernel, C=self.config['c']).fit(train, trainLabel)

    def crossValidation(self, rolling):
        re = []
        for C in [0.7, 1, 1.3, 2, 3, 7, 10, 13, 20, 100, 1000, 10000]:
            self.config['c'] = C
            re.append([C, 
                       self.crossValidationScore(
                           self.cvAgg(isRolling=rolling)
                           )
                       ]
                      )
        return re

def forecastAndPlot(mlFactory, feature, algoName, rolling=True, drAlgo=''):
    algo = mlFactory.create(algoName, feature)
    if rolling:
        re, reTrain = algo.rollingCrossValidation()
    else:
        re, reTrain = algo.nonRollingTrainTestError()
    
    outOfSampleAccuracy = float(np.mean(np.array(re)[:, 1]))
    
    plt.plot(np.array(re)[:, 0], np.array(re)[:, 1], label='Test')
    plt.plot(np.array(reTrain)[:, 0], np.array(reTrain)[:, 1], label='Train')
    
    plt.xlabel('Number of training period')
    plt.ylabel('Testing Accuracy')
    plt.legend()
    plt.title('Rolling cross validation accuracy {algo} {dr} {a}'.format(
        algo=algoName, dr=drAlgo, a=floor(outOfSampleAccuracy * 100) / 100)
    )
    plt.grid()
    plt.show()

def cvAndPlot(mlFactory, feature, algoName):
    algo = mlFactory.create(algoName, feature)
    re = algo.crossValidation(True)
    plt.plot(np.array(re)[:, 0], np.array(re)[:, 1], label='Test')

    plt.xlabel('Hyper prameters')
    plt.ylabel('Testing Accuracy')
    plt.legend()
    plt.title('Cross validation accuracy {algo}'.format(algo=algoName))
    plt.show()

def testRollingValidation():
    sd = datetime.date(2011, 10, 28)
    ed = datetime.date(2016, 10, 28)
    swapMLData = SwapMLData(sd=sd, ed=ed)
    swapMLData.loadData()
    swapMLData.processData()
    
    feature = Features(swapMLData)
    feature.combineFeatures()
    
    mlFactory = SwapMLFactory()
    for algoName, _ in mlFactory.mlRegistry.items():
        forecastAndPlot(mlFactory, feature, algoName)


class CreditFeatures:
    
    def __init__(self):
        self.data = pd.read_csv('german.data-numeric', sep='\s\s+|,', header=None)
        self.data = self.data.dropna()
        self.X = np.array(self.data.iloc[:, :23]).astype(float)
        self.features = self.data.iloc[:, :23]
        self.y = np.array(self.data.iloc[:, 24].astype(int)) - 1
        self.labels = self.data.iloc[:, 24]
    
    def enrichLabelKMeans(self, X, k=2):
        kmeansCluster = KMeans(n_clusters=k).fit(self.X)
        y_pred = np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
        self.features['label'] = y_pred
        return

    def enrichLabelGMM(self, X, k=2):
        estimator = GaussianMixture(n_components=k,
                   covariance_type='full', max_iter=20, random_state=0).fit(self.X)
        y_pred = estimator.predict(self.X)
        self.features['label'] = y_pred
        return

    def doPCA(self, n=3, k=2, isKmeans=True):
        X = PCA(n_components=n).fit(self.X).transform(self.X)
        if isKmeans:
            self.enrichLabelKMeans(X, k)
        else:
            self.enrichLabelGMM(X, k)
        return
        
    def doICA(self, k=2, isKmeans=True):
        ica = decomposition.FastICA(n_components=3)
        X = ica.fit(self.X, self.y).transform(self.X)
        if isKmeans:
            self.enrichLabelKMeans(X, k)
        else:
            self.enrichLabelGMM(X, k)
        return
    
    def doRP(self, k=2, isKmeans=True):
        random = random_projection.GaussianRandomProjection(n_components=3)
        X = random.fit(self.X).transform(self.X)
        if isKmeans:
            self.enrichLabelKMeans(X, k)
        else:
            self.enrichLabelGMM(X, k)
        return
    
class Kmeans:
    
    def __init__(self, data):
        self.data = data
    
    def kmeans(self, k=2):
        kmeansCluster = KMeans(n_clusters=2).fit(self.data.X)
        y_pred = np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
        score = sm.accuracy_score(self.data.y, y_pred)
        print(k, score)
        print(sm.confusion_matrix(self.data.y, y_pred))
        return score
    
    def pca(self):
        pc = PCA().fit(self.data.X)
        var_ratio = pc.explained_variance_ratio_
        plt.bar(var_ratio, list(range(var_ratio.shape[0])))
        plt.title('PCA Explained Variance vs No Components')
        plt.show()
        X = PCA(n_components=3).fit(self.data.X).transform(self.data.X)
        kmeansCluster = KMeans(n_clusters=2).fit(X)
        y_pred = np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
        score = sm.accuracy_score(self.data.y, y_pred)
        print('pca', score)
        print(sm.confusion_matrix(self.data.y, y_pred))
        return score

    def ica(self):
        ica = decomposition.FastICA(n_components=3)
        X = ica.fit(self.data.X, self.data.y).transform(self.data.X)
        kmeansCluster = KMeans(n_clusters=2).fit(X)
        y_pred = np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
        score = sm.accuracy_score(self.data.y, y_pred)
        print(sm.confusion_matrix(self.data.y, y_pred))
        print('ica', score)
        return score

    def random(self):
        random = random_projection.GaussianRandomProjection(n_components=3)
        X = random.fit(self.data.X).transform(self.data.X)
        kmeansCluster = KMeans(n_clusters=2).fit(X)
        y_pred = np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
        score = sm.accuracy_score(self.data.y, y_pred)
        print('random', score)
        print(sm.confusion_matrix(self.data.y, y_pred))
        return score

class GMM:
    
    def __init__(self, data):
        self.data = data
    
    def gmm(self, n=2):
        estimator = GaussianMixture(n_components=2,
                   covariance_type='full', max_iter=20, random_state=0).fit(self.data.X)
        y_pred = estimator.predict(self.data.X)
        score = sm.accuracy_score(self.data.y, y_pred)
        print(score)
        print(sm.confusion_matrix(self.data.y, y_pred))
        return score
    
    def pca(self):
        pc = PCA().fit(self.data.X)
        var_ratio = pc.explained_variance_ratio_
        plt.bar(var_ratio, list(range(var_ratio.shape[0])))
        plt.title('PCA Explained Variance vs No Components')
        plt.show()
        X = PCA(n_components=3).fit(self.data.X).transform(self.data.X)
        estimator = GaussianMixture(n_components=2,
                   covariance_type='full', max_iter=20, random_state=0).fit(X)
        y_pred = estimator.predict(self.data.X)
        score = sm.accuracy_score(self.data.y, y_pred)
        print('pca', score)
        print(sm.confusion_matrix(self.data.y, y_pred))
        return score

    def ica(self):
        ica = decomposition.FastICA(n_components=3)
        X = ica.fit(self.data.X, self.data.y).transform(self.data.X)
        kmeansCluster = KMeans(n_clusters=2).fit(X)
        y_pred = np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
        score = sm.accuracy_score(self.data.y, y_pred)
        print(sm.confusion_matrix(self.data.y, y_pred))
        print('ica', score)
        return score

    def random(self):
        random = random_projection.GaussianRandomProjection(n_components=3)
        X = random.fit(self.data.X).transform(self.data.X)
        kmeansCluster = KMeans(n_clusters=2).fit(X)
        y_pred = np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
        score = sm.accuracy_score(self.data.y, y_pred)
        print('random', score)
        print(sm.confusion_matrix(self.data.y, y_pred))
        return score

def rawCluster(X, y):
    estimator = GaussianMixture(n_components=2,
               covariance_type='full', max_iter=20, random_state=0).fit(X)
    y_pred = 1 - estimator.predict(X)
    score = sm.accuracy_score(y, y_pred)
    print(score)
    print(sm.confusion_matrix(y, y_pred))

    kmeansCluster = KMeans(n_clusters=2).fit(X)
    y_pred = 1 - np.choose(kmeansCluster.labels_, [0, 1]).astype(int)
    score = sm.accuracy_score(y, y_pred)
    print(score)
    print(sm.confusion_matrix(y, y_pred))
    return

def rawClusterIris(X, y):
    estimator = GaussianMixture(n_components=3,
               covariance_type='full', max_iter=20, random_state=0).fit(X)
    y_pred = estimator.predict(X)
    score = sm.accuracy_score(y, y_pred)
    print(score)
    print(sm.confusion_matrix(y, y_pred))

    kmeansCluster = KMeans(n_clusters=3).fit(X)
    y_pred = np.choose(kmeansCluster.labels_, [2, 0, 1]).astype(int)
    score = sm.accuracy_score(y, kmeansCluster.labels_)
    print(score)
    print(sm.confusion_matrix(y, y_pred))
    return


def testAccuracy():
    feature = CreditFeatures()
    
    mlFactory = SwapMLFactory()
    # for algoName, _ in mlFactory.mlRegistry.items():
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False)
    
    feature.doPCA(3, 2, True)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='KMeans PCA')
    feature.doPCA(3, 2, False)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='EM PCA')

    feature.doICA(2, True)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='KMeans ICA')
    feature.doICA(2, False)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='EM ICA')

    feature.doRP(2, True)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='KMeans RP')
    feature.doRP(2, False)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='EM RP')

    feature.doICA(2, True)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='KMeans LDA')
    feature.doICA(2, False)
    forecastAndPlot(mlFactory, feature, 'NeuralNet', rolling=False, drAlgo='EM LDA')
    
    model = KMeans()
    vis = KElbowVisualizer(model, k=(4,12))
    vis.fit(feature.X)
    vis.show()
    
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data)
    x.columns = ['Length','Width','PL','PW']
 
    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']

    colormap = np.array(['red', 'green', 'organge'])
    plt.scatter(x.Length, x.PL, c=colormap[y.Targets], s=40)
    plt.title('Length')
    
    model = KMeans()
    
def testKmeans():
    pass
    
if __name__ == '__main__':
    testAccuracy()    
    
    