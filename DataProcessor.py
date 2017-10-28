from __future__ import division

import sys
import numpy as np
import time
from collections import defaultdict

def map_data(filename, feature2index):
    data = [] # list of (vecx, y) pairs
    target = []
    dimension = len(feature2index)
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:-1]): # last one is target
            if (i, fv) in feature2index: # ignore unobserved features
                feat_vec[feature2index[i, fv]] = 1

        data.append(feat_vec)
        
        if features[-1] == ">50K":
            y = 1
        else:
            y = -1

        target.append(y)

    return data, target

def test_data(filename, feature2index):
    data = [] # list of (vecx, y) pairs
    target = []
    dimension = len(feature2index)
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:-1]): # last one is target
            if (i, fv) in feature2index: # ignore unobserved features
                feat_vec[feature2index[i, fv]] = 1

        data.append(feat_vec)

    return data

def create_feature_map(train_file):

    column_values = defaultdict(set)
    for line in open(train_file):
        line = line.strip()
        features = line.split(", ")[:-1] # last field is target
        for i, fv in enumerate(features):
            column_values[i].add(fv)

    feature2index = {}
    for i, values in column_values.items():
        for v in values:            
            feature2index[i, v] = len(feature2index)

    dimension = len(feature2index)
    print ("dimensionality: ", dimension)
    return feature2index
