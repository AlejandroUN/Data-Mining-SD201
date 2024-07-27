from typing import List

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 father_threshold: float = -1,
                 min_split_points: int = 1,
                 father_points: PointSet = PointSet([[]],[],[])):


        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        self.points = PointSet(features, labels, types)
        self.h = h
        self.father_threshold = father_threshold
        self.min_split_points = min_split_points
        self.father_points = father_points
        self.isLeaf = False
        count0 = 0
        count1 = 0
        for x in labels:
            if x == 0:
                count0 += 1 
            else:
                count1 += 1 
        if (len(self.points.features) < (2*self.min_split_points)):
            self.isLeaf = True
        if (count0==len(labels) or count1==len(labels)) or h==1:
            self.isLeaf = True
        self.childLeftFeatures = []
        self.childLeftLabels = []
        self.childRightFeatures = []
        self.childRightLabels = []
        self.featureIndex = 0
        [self.featureIndex,gainValue] = self.points.get_best_gain(self.min_split_points)
        if self.featureIndex == -1:
            self.isLeaf = True
        if self.isLeaf == False:
            if self.points.types[self.featureIndex] == FeaturesTypes.REAL:
                for dataPoint in range(len(features)):
                    if features[dataPoint][self.featureIndex] < self.points.get_best_threshold():
                        self.childLeftFeatures.append(features[dataPoint])
                        self.childLeftLabels.append(labels[dataPoint])
                    else:
                        self.childRightFeatures.append(features[dataPoint])
                        self.childRightLabels.append(labels[dataPoint])
                self.leftChild = Tree(self.childLeftFeatures, self.childLeftLabels, types,h-1,self.points.get_best_threshold(),self.min_split_points, self.points)
                self.rightChild = Tree(self.childRightFeatures, self.childRightLabels, types,h-1,self.points.get_best_threshold(),self.min_split_points, self.points)
            elif self.points.types[self.featureIndex] == FeaturesTypes.CLASSES:
                for dataPoint in range(len(features)):
                    if features[dataPoint][self.featureIndex] == self.points.valuesOfTheBestCategorical[self.points.bestCategoricalIndex]:
                        self.childLeftFeatures.append(features[dataPoint])
                        self.childLeftLabels.append(labels[dataPoint])
                    else:
                        self.childRightFeatures.append(features[dataPoint])
                        self.childRightLabels.append(labels[dataPoint])
                self.leftChild = Tree(self.childLeftFeatures, self.childLeftLabels, types,h-1,min_split_points=self.min_split_points, father_points=self.points)
                self.rightChild = Tree(self.childRightFeatures, self.childRightLabels, types,h-1,min_split_points=self.min_split_points, father_points=self.points)
            else:
                for dataPoint in range(len(features)):
                    if features[dataPoint][self.featureIndex] == 0.0:
                        self.childLeftFeatures.append(features[dataPoint])
                        self.childLeftLabels.append(labels[dataPoint])
                    if features[dataPoint][self.featureIndex] == 1.0:
                        self.childRightFeatures.append(features[dataPoint])
                        self.childRightLabels.append(labels[dataPoint])
                self.leftChild = Tree(self.childLeftFeatures, self.childLeftLabels, types,h-1,min_split_points=self.min_split_points, father_points=self.points)
                self.rightChild = Tree(self.childRightFeatures, self.childRightLabels, types,h-1,min_split_points=self.min_split_points, father_points=self.points)
        else:
                [self.featureIndex,gainValue] = self.points.get_best_gain(self.min_split_points)
                if self.points.types[self.featureIndex] == FeaturesTypes.REAL:
                    if self.featureIndex == -1:
                        thresholdToUse = self.father_threshold
                    else:
                        thresholdToUse = self.points.get_best_threshold()
                    for dataPoint in range(len(features)):
                        if features[dataPoint][self.featureIndex] < thresholdToUse:
                            self.childLeftFeatures.append(features[dataPoint])
                            self.childLeftLabels.append(labels[dataPoint])
                        else:
                            self.childRightFeatures.append(features[dataPoint])
                            self.childRightLabels.append(labels[dataPoint])
                    self.leftChild = PointSet(self.childLeftFeatures, self.childLeftLabels, types, thresholdToUse)
                    self.rightChild = PointSet(self.childRightFeatures, self.childRightLabels, types, thresholdToUse)
                elif self.featureIndex != -1:
                    if self.points.types[self.featureIndex] == FeaturesTypes.CLASSES:
                        for dataPoint in range(len(features)):
                            if features[dataPoint][self.featureIndex] == self.points.valuesOfTheBestCategorical[self.points.bestCategoricalIndex]:
                                self.childLeftFeatures.append(features[dataPoint])
                                self.childLeftLabels.append(labels[dataPoint])
                            else:
                                self.childRightFeatures.append(features[dataPoint])
                                self.childRightLabels.append(labels[dataPoint])
                        self.leftChild = PointSet(self.childLeftFeatures, self.childLeftLabels, types)
                        self.rightChild = PointSet(self.childRightFeatures, self.childRightLabels, types)
                    else:
                        for dataPoint in range(len(features)):
                            if features[dataPoint][self.featureIndex] == 0.0:
                                self.childLeftFeatures.append(features[dataPoint])
                                self.childLeftLabels.append(labels[dataPoint])
                            if features[dataPoint][self.featureIndex] == 1.0:
                                self.childRightFeatures.append(features[dataPoint])
                                self.childRightLabels.append(labels[dataPoint])
                        self.leftChild = PointSet(self.childLeftFeatures, self.childLeftLabels, types)
                        self.rightChild = PointSet(self.childRightFeatures, self.childRightLabels, types)

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        if self.isLeaf:
            if (len(self.points.features) < (2*self.min_split_points)):
                trueCount = 0
                falseCount = 0
                for point in range(len(self.points.labels)):
                    if self.points.labels[point] == True:
                        trueCount += 1
                    else:
                        falseCount += 1
                if trueCount > falseCount:
                    return True
                else:
                    return False
            trueCount = 0
            falseCount = 0
            for point in range(len(self.points.labels)):
                if self.points.labels[point] == True:
                    trueCount += 1
                else:
                    falseCount += 1
            if ((trueCount) == len(self.points.labels)):
                return True
            elif ((falseCount) == len(self.points.labels)):
                return False

            if self.points.types[self.featureIndex] == FeaturesTypes.REAL:
			#now, if the get_best_gini = -1 use the threhold of the father
                if self.featureIndex == -1:
                    if self.father_threshold != -1:
                        thresholdToUse = self.father_threshold
                    elif self.points.father_threshold != -1:
                        thresholdToUse = self.points.father_threshold
                else:
                    thresholdToUse = self.points.get_best_threshold()
                if features[self.featureIndex] < thresholdToUse:
                    trueCount = 0
                    falseCount = 0
                    for point in range(len(self.childLeftLabels)):
                        if self.childLeftLabels[point] == True:
                            trueCount += 1
                        else:
                            falseCount += 1
                    if trueCount > falseCount:
                        return True
                    else:
                        return False
                else:
                    trueCount = 0
                    falseCount = 0
                    for point in range(len(self.childRightLabels)):
                        if self.childRightLabels[point] == True:
                            trueCount += 1
                        else:
                            falseCount += 1
                    if trueCount > falseCount:
                        return True
                    else:
                        return False
			#now, if the get_best_gini = -1 use the points and the index of the father

            if self.featureIndex == -1:
                indexToUse = self.father_points.get_best_gain()[0]
            else:
                indexToUse = self.featureIndex
            if self.points.types[self.featureIndex] == FeaturesTypes.CLASSES:
                if self.featureIndex == -1:
                    categoricalToUse = self.father_points.valuesOfTheBestCategorical[self.father_points.bestCategoricalIndex]
                else:
                    categoricalToUse = self.points.valuesOfTheBestCategorical[self.points.bestCategoricalIndex]
                if features[indexToUse] == categoricalToUse:
                    trueCount = 0
                    falseCount = 0
                    for point in range(len(self.childLeftLabels)):
                        if self.childLeftLabels[point] == True:
                            trueCount += 1
                        else:
                            falseCount += 1
                    if trueCount > falseCount:
                        return True
                    else:
                        return False
                else:
                    trueCount = 0
                    falseCount = 0
                    for point in range(len(self.childRightLabels)):
                        if self.childRightLabels[point] == True:
                            trueCount += 1
                        else:
                            falseCount += 1
                    if trueCount > falseCount:
                        return True
                    else:
                        return False
            else:
                if features[indexToUse] == 0.0:
                    trueCount = 0
                    falseCount = 0
                    for point in range(len(self.childLeftLabels)):
                        if self.childLeftLabels[point] == True:
                            trueCount += 1
                        else:
                            falseCount += 1
                    if trueCount > falseCount:
                        return True
                    else:
                        return False
                else:
                    trueCount = 0
                    falseCount = 0
                    for point in range(len(self.childRightLabels)):
                        if self.childRightLabels[point] == True:
                            trueCount += 1
                        else:
                            falseCount += 1
                    if trueCount > falseCount:
                        return True
                    else:
                        return False
        else:
            if self.points.types[self.featureIndex] == FeaturesTypes.REAL:
                if features[self.featureIndex] < self.points.get_best_threshold():
                    #see the most probable value in that child
                    return self.leftChild.decide(features)
                else:
                    #see the most probable value in that child
                    return self.rightChild.decide(features)
            elif self.points.types[self.featureIndex] == FeaturesTypes.CLASSES:
                if features[self.featureIndex] == self.points.valuesOfTheBestCategorical[self.points.bestCategoricalIndex]:
                    return self.leftChild.decide(features)
                else:
                    return self.rightChild.decide(features)
            else:
                if features[self.featureIndex] == 0.0:
                    #see the most probable value in that child
                    return self.leftChild.decide(features)
                else:
                    #see the most probable value in that child
                    return self.rightChild.decide(features)

