#!/usr/bin/python3
# -*- coding: utf-8 -*-
# *****************************************************************************/
# * Copyright 2021 UCR CS205, Artificial Intelligence. All Rights Reserved.
# * Closed source repository. Do not share any content without permission written from Rogelio Macedo.
# * Authors: Rogelio Macedo
# * Template Credit: Joseph Tarango
# *****************************************************************************/

"""
Resources
- https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
- https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
- https://www.geeksforgeeks.org/filter-in-python/
- https://www.programiz.com/python-programming/generator
- https://github.com/rmace001/Feature-Selection/
- https://stackoverflow.com/questions/43367001/how-to-calculate-euclidean-distance-between-pair-of-rows-of-a-numpy-array

Usage: $: python3 FeatureSelect2.0.py --filesize small --mode 1 --debug
Args:
    --filesize: ['small' | 'large']
    --mode: [1 | 2 | 3]
    --debug: stores True
"""


import json
import optparse
import os
import datetime
import traceback
import numpy as np


def leave1OutCrossVal(curData=None, classes=None):
    correct: np.double = 0
    dataReshaped = curData.reshape(curData.shape[0], 1, curData.shape[1])
    distances = np.sqrt(np.einsum('ijk, ijk->ij', curData-dataReshaped, curData-dataReshaped))
    for i in range(distances.shape[0]):
        currentRow = distances[i, :]
        indices = ((j, currentRow[j]) for j in range(currentRow.shape[0]))
        top = sorted(indices, key=lambda x: x[1])[1]
        if classes[i] == classes[top[0]]:
            correct += 1.0
    return correct/distances.shape[0]


def FeatureSelection(data=None, mode=None, debug=None):
    currentFeatureSet = set()
    bestOverall = set()
    globalBest: np.double = 0.0
    levelBest:  np.double = 0.0
    # accuracies = list()
    for i in range(1, data.shape[1]):
        accuracy: np.double = 0.0
        bestSoFarAccuracy: np.double = 0.0
        feature2AddAtCurrentLevel: int = -1

        for j in range(1, data.shape[1]):
            if j not in currentFeatureSet:
                tempFeatureSet = set(currentFeatureSet)
                tempFeatureSet.add(j)
                print(f'\t\tUsing Features: {tempFeatureSet}')
                zeroedFeatures = np.zeros(shape=(data.shape[0], data.shape[1]))
                zeroedFeatures[:, list(tempFeatureSet)] = data[:, list(tempFeatureSet)]
                accuracy: np.double = leave1OutCrossVal(curData=zeroedFeatures[:, 1:],
                                                        classes=data[:, 0].astype('int8'))
                print(f'Accuracy is {round(accuracy*100, 3)}%')
                # accuracies.append(accuracy)

            if accuracy > bestSoFarAccuracy:
                bestSoFarAccuracy = accuracy
                feature2AddAtCurrentLevel = j
        if levelBest < bestSoFarAccuracy:
            levelBest = bestSoFarAccuracy
            if globalBest < levelBest:
                globalBest = levelBest
                bestOverall = set(currentFeatureSet)
                bestOverall.add(feature2AddAtCurrentLevel)
        else:
            if i < data.shape[1] - 2:
                print('(Warning, Accuracy has decreased! Continuing search in case of local maxima)')
        if feature2AddAtCurrentLevel not in currentFeatureSet:
            currentFeatureSet.add(feature2AddAtCurrentLevel)
            if i < data.shape[1] - 2:
                print(f'Feature Set: {currentFeatureSet} was best, accuracy is {round(bestSoFarAccuracy*100, 3)}%\n')
    print(f'\nFinished search! The best feature subset is {bestOverall}, yeilding an accuracy of {globalBest*100}%.')
    return bestOverall

def CLI():
    ##############################################
    # Main function, Options
    ##############################################
    parser = optparse.OptionParser()
    parser.add_option("--example", action='store_true', dest='example', default=False,
                      help='Show command execution example.')
    parser.add_option("--debug", action='store_true', dest='debug', default=True, help='Debug mode.')
    parser.add_option("--filesize", dest='filesize', default='small', help="Operate on \'small\' or \'large\' dataset")  # 'small dataset'
    parser.add_option("--mode", dest='mode', default='1', help="1: Foward Search, 2: Backward Search, 3: Rogelio\'s Special Search")
    (options, args) = parser.parse_args()
    try:
        main(options=options)
    except Exception as errorMain:
        print("Fail End Process: {0}".format(errorMain))
        traceback.print_exc()
    return


def filterWhiteSpace(elem):
    return elem != ''


def gen(iterableItem):
    for item in iterableItem:
        yield item


def main(options=None):
    print("Welcome to Rogelio\'s Feature Selection Algorithm.")

    if str.lower(options.filesize) == 'small':
        filename = 'CS205_small_testdata__37.txt'
    elif str.lower(options.filesize) == 'large':
        filename = 'CS205_large_testdata__4.txt'
    else:
        print(f'Invalid filesize chosen: {options.filesize}. Valid filesizes: [small | large]')

    print(f'File to test: {filename}')
    print('Algorithm options:')
    print('\t1) Forward Selection')
    print('\t2) Backward Selection')
    print('\t3) Rogelio\'s Special Search')

    if options.mode == '1':
        pass
    elif options.mode == '2':
        pass
    elif options.mode == '3':
        pass
    else:
        print(f'Invalid mode chosen: {options.mode}. Valid modes: [1 | 2 | 3]')

    print(f'Mode option chosen: {options.mode}')

    fileread = os.path.abspath(os.path.join(os.getcwd(), filename))
    # lines = (line for line in open(fileread))
    # linesAsLists = (s.rstrip().split(' ') for s in lines)
    # lineSplits = (gen(filter(filterWhiteSpace, value)) for value in linesAsLists)
    # lineFeatures = (value for value in lineSplits if value != '')
    # for feats in lineFeatures:
    #     for feat in feats:
    #         pass
    #         print(feat)
    #         print(round(float(feat), 3))
    arr = np.loadtxt(fname=fileread)
    bestFeatures = FeatureSelection(data=arr, mode=options.mode, debug=options.debug)
    return



if __name__ == "__main__":
    pStart = datetime.datetime.now()
    CLI()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop - pStart))
