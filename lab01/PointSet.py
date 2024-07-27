from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes], father_threshold: float = -1):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.father_threshold = father_threshold
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.theBestIsCategorical = False
        self.bestCategoricalIndex = -1
        self.valuesOfTheBestCategorical = []
        self.theBestIsContinuous = False
        self.bestContinuousIndex = -1
        self.valuesOfTheBestContinuous = []
        self.indexBestfeature = -1
        self.wasBestGiniCalled = -1

    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        sumTrue = 0
        sumFalse = 0
        for point in self.labels:
            if point == True:
                sumTrue += 1
            elif point == False:
                sumFalse += 1
        probClass1 = float((sumTrue/(sumTrue+sumFalse))**2)
        probClass2 = float((sumFalse/(sumTrue+sumFalse))**2)
        gini = 1 - (probClass1 + probClass2)
        return gini
        #raise NotImplementedError('Please implement this function for Question 1')

#for this method i'm gonna return 
#[] list of possible values for the categorical 
#the ginigain of the max
#the index of the categoricalValue in []
    def calculate_best_gini_categorical(self,features,indexCategorical,min_split_points: int = -1):
        #first let's see how many different values have the categorical feature
        rightChild = 0
        leftChild = 0
        giniOriginal = self.get_gini()
        categoricalValues = []
        for point in range(len(features)):
            if features[point][indexCategorical] in categoricalValues:
                continue
            else:
                categoricalValues.append(features[point][indexCategorical])
        #once we got them let's calculate the gini for every possible combination
        ginis = np.zeros((len(categoricalValues)))
        max = 0
        best = -1
        for categoricalValueIndex in range(len(categoricalValues)):
                leftChild = []
                rightChild = []
                for dataPoint in range(len(features)):
                    if features[dataPoint][indexCategorical] == categoricalValues[categoricalValueIndex]:
                        leftChild.append(self.labels[dataPoint])
                    else:
                        rightChild.append(self.labels[dataPoint])
                #if there is a feature with the same value for all points return None
                if (len(leftChild) == len(features)) or (len(rightChild) == len(features)):
                    continue
				#evaluate new min condition
                if (len(leftChild) < (min_split_points)) or (len(rightChild) < (min_split_points)):
                    continue
			    #just in case here is the none
                    #return None
    
                #calculate gini for each child
			    #calculate it for left child
                sumTrue = 0
                sumFalse = 0
                for labelC in leftChild:
                    if labelC == True:
                        sumTrue += 1
                    elif labelC == False:
                        sumFalse += 1
                probClass1 = float((sumTrue/(sumTrue+sumFalse))**2)
                probClass2 = float((sumFalse/(sumTrue+sumFalse))**2)
                giniLChild = 1 - (probClass1 + probClass2)
			    #calculate it for right child
			    #calculate it for left child
                sumTrue = 0
                sumFalse = 0
                for labelC in rightChild:
                    if labelC == True:
                        sumTrue += 1
                    elif labelC == False:
                        sumFalse += 1
                probClass1 = float((sumTrue/(sumTrue+sumFalse))**2)
                probClass2 = float((sumFalse/(sumTrue+sumFalse))**2)
                giniRChild = 1 - (probClass1 + probClass2)
                #calculate gini split
                giniSplit = ((len(leftChild)*giniLChild)/(len(leftChild)+len(rightChild))) + ((len(rightChild)*giniRChild)/(len(leftChild)+len(rightChild)))
                #calculate gain
                giniGain = giniOriginal - giniSplit
                ginis[categoricalValueIndex] = giniGain
                if giniGain > max:
                    max = giniGain
                    best = categoricalValueIndex
        return [categoricalValues,max,best]

#for this method i'm gonna return 
#[] list of possible values for the continuous 
#the ginigain of the max
#the index of the continuousValue in []
    def calculate_best_gini_continuous(self,features,indexContinuous,min_split_points: int = -1):
        #first let's see how many different values have the continuous feature
        rightChild = 0
        leftChild = 0
        giniOriginal = self.get_gini()
        continuousValues = []
        for point in range(len(features)):
            if features[point][indexContinuous] in continuousValues:
                continue
            else:
                continuousValues.append(features[point][indexContinuous])
        #once we got them let's calculate the gini for every possible combination
        ginis = np.zeros((len(continuousValues)))
        max = 0
        best = -1
        for continuousValueIndex in range(len(continuousValues)):
                leftChild = []
                rightChild = []
                for dataPoint in range(len(features)):
                    if features[dataPoint][indexContinuous] < continuousValues[continuousValueIndex]:
                        leftChild.append(self.labels[dataPoint])
                    else:
                        rightChild.append(self.labels[dataPoint])
                #if there is a feature with the same value for all points return None
                if (len(leftChild) == len(features)) or (len(rightChild) == len(features)):
                    continue
				#evaluate new min condition
                if (len(leftChild) < (min_split_points)) or (len(rightChild) < (min_split_points)):
                    continue
			    #just in case here is the none
                    #return None
    
                #calculate gini for each child
			    #calculate it for left child
                sumTrue = 0
                sumFalse = 0
                for labelC in leftChild:
                    if labelC == True:
                        sumTrue += 1
                    elif labelC == False:
                        sumFalse += 1
                probClass1 = float((sumTrue/(sumTrue+sumFalse))**2)
                probClass2 = float((sumFalse/(sumTrue+sumFalse))**2)
                giniLChild = 1 - (probClass1 + probClass2)
			    #calculate it for right child
			    #calculate it for left child
                sumTrue = 0
                sumFalse = 0
                for labelC in rightChild:
                    if labelC == True:
                        sumTrue += 1
                    elif labelC == False:
                        sumFalse += 1
                probClass1 = float((sumTrue/(sumTrue+sumFalse))**2)
                probClass2 = float((sumFalse/(sumTrue+sumFalse))**2)
                giniRChild = 1 - (probClass1 + probClass2)
                #calculate gini split
                giniSplit = ((len(leftChild)*giniLChild)/(len(leftChild)+len(rightChild))) + ((len(rightChild)*giniRChild)/(len(leftChild)+len(rightChild)))
                #calculate gain
                giniGain = giniOriginal - giniSplit
                ginis[continuousValueIndex] = giniGain
                if giniGain > max:
                    max = giniGain
                    best = continuousValueIndex
        return [continuousValues,max,best]


    def get_best_gain(self,min_split_points: int = -1) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        self.wasBestGiniCalled = 1
                                    #calculate original gini

        giniOriginal = self.get_gini()
                     
					#Calculate giniGain for each feature

        rightChild = 0
        leftChild = 0
        ginis = np.zeros((len(self.features[0])))
        max = 0
        best = -1
        for featureNumber in range(len(self.features[0])):
			#calculate ginigain now
            leftChild = []
            rightChild = []
            if self.types[featureNumber] == FeaturesTypes.REAL:
                continuousArrayValues = []
                maxGiniContinuous = -1
                mexGiniIndexContinuous = -1
                tempForAllValues = []
                tempForAllValues = self.calculate_best_gini_continuous(self.features, featureNumber,min_split_points)
                continuousArrayValues = tempForAllValues[0]
                maxGiniContinuous = tempForAllValues[1]
                mexGiniIndexContinuous = tempForAllValues[2]
                if mexGiniIndexContinuous == -1:
                    continue
                ginis[featureNumber] = maxGiniContinuous
                if maxGiniContinuous > max:
                    max = maxGiniContinuous
                    best = featureNumber
                    self.theBestIsContinuous = True
                    self.bestContinuousIndex = mexGiniIndexContinuous
                    self.valuesOfTheBestContinuous = continuousArrayValues
                    self.theBestIsCategorical = False
                    self.bestCategoricalIndex = -1
                    self.valuesOfTheBestCategorical = []
            elif self.types[featureNumber] == FeaturesTypes.CLASSES:
                categoricalArrayValues = []
                maxGiniCategorical = -1
                mexGiniIndexCategorical = -1
                tempForAllValues = []
                tempForAllValues = self.calculate_best_gini_categorical(self.features, featureNumber,min_split_points)
                categoricalArrayValues = tempForAllValues[0]
                maxGiniCategorical = tempForAllValues[1]
                mexGiniIndexCategorical = tempForAllValues[2]
                if mexGiniIndexCategorical == -1:
                    continue
                ginis[featureNumber] = maxGiniCategorical
                if maxGiniCategorical > max:
                    max = maxGiniCategorical
                    best = featureNumber
                    self.theBestIsCategorical = True
                    self.bestCategoricalIndex = mexGiniIndexCategorical
                    self.valuesOfTheBestCategorical = categoricalArrayValues
                    self.theBestIsContinuous = False
                    self.bestContinuousIndex = -1
                    self.valuesOfTheBestContinuous = []

            else:
                for dataPoint in range(len(self.features)):
                    if self.features[dataPoint][featureNumber] == 0.0:
                        leftChild.append(self.labels[dataPoint])
                    else:
                        rightChild.append(self.labels[dataPoint])
                #if there is a feature with the same value for all points return None
                if (len(leftChild) == len(self.features)) or (len(rightChild) == len(self.features)):
                    continue
				#evaluate new min condition
                if (len(leftChild) < (min_split_points)) or (len(rightChild) < (min_split_points)):
                    continue
			    #just in case here is the none
                    #return None
    
                #calculate gini for each child
			    #calculate it for left child
                sumTrue = 0
                sumFalse = 0
                for labelC in leftChild:
                    if labelC == True:
                        sumTrue += 1
                    elif labelC == False:
                        sumFalse += 1
                probClass1 = float((sumTrue/(sumTrue+sumFalse))**2)
                probClass2 = float((sumFalse/(sumTrue+sumFalse))**2)
                giniLChild = 1 - (probClass1 + probClass2)
			    #calculate it for right child
			    #calculate it for left child
                sumTrue = 0
                sumFalse = 0
                for labelC in rightChild:
                    if labelC == True:
                        sumTrue += 1
                    elif labelC == False:
                        sumFalse += 1
                probClass1 = float((sumTrue/(sumTrue+sumFalse))**2)
                probClass2 = float((sumFalse/(sumTrue+sumFalse))**2)
                giniRChild = 1 - (probClass1 + probClass2)
                #calculate gini split
                giniSplit = ((len(leftChild)*giniLChild)/(len(leftChild)+len(rightChild))) + ((len(rightChild)*giniRChild)/(len(leftChild)+len(rightChild)))
                #calculate gain
                giniGain = giniOriginal - giniSplit
                ginis[featureNumber] = giniGain
                if giniGain > max:
                    max = giniGain
                    best = featureNumber
                    self.theBestIsCategorical = False
                    self.bestCategoricalIndex = -1
                    self.valuesOfTheBestCategorical = []
                    self.theBestIsContinuous = False
                    self.bestContinuousIndex = -1
                    self.valuesOfTheBestContinuous = []
        self.indexBestfeature = best
        return [best,max]

    def get_best_threshold(self) -> float:
        if self.wasBestGiniCalled == -1:
            raise Exception('get best gini method was not called')
        if self.types[self.indexBestfeature] == FeaturesTypes.BOOLEAN:
            return None
        elif self.types[self.indexBestfeature] == FeaturesTypes.CLASSES:
            return self.valuesOfTheBestCategorical[self.bestCategoricalIndex]
        elif self.types[self.indexBestfeature] == FeaturesTypes.REAL:
            leftChild = []
            rightChild = []
            for dataPoint in range(len(self.features)):
                if self.features[dataPoint][self.indexBestfeature] < self.valuesOfTheBestContinuous[self.bestContinuousIndex]:
                    leftChild.append(self.features[dataPoint][self.indexBestfeature])
                else:
                    rightChild.append(self.features[dataPoint][self.indexBestfeature])
            return ((max(leftChild)+min(rightChild))/2)