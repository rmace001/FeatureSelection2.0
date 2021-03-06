#!/usr/bin/python3
# -*- coding: utf-8 -*-
# *****************************************************************************/
# * Copyright 2021 UCR CS205, Artificial Intelligence. All Rights Reserved.
# * Closed source repository. Do not share any content without permission written from Rogelio Macedo.
# * Authors: Rogelio Macedo
# * Template Credit: Joseph Tarango
# *****************************************************************************/

"""

Usage: $: python3 FeatureSelect2.0.py --filesize small --mode 1 --debug
Args:
    --filesize: ['small' | 'large' | 'None']
    --mode: [1 | 2 | 3]
    --debug: stores True

"""

import optparse
import os
import datetime
import traceback
import numpy as np
import pandas as pd
import math


def leave1OutCrossVal(curData=None, classes=None):
    correct: np.double = 0
    dataReshaped       = curData.reshape(curData.shape[0], 1, curData.shape[1])

    #                    omit sqrt operation to optimize distance calculation
    distances          = np.einsum('ijk, ijk->ij', curData-dataReshaped, curData-dataReshaped)

    for i in range(distances.shape[0]):
        currentRow     = distances[i, :]
        indices        = ((j, currentRow[j]) for j in range(currentRow.shape[0]))
        curmin         = (0, math.inf)

        for item in indices:
            if item[0] != i:
                if item[1] < curmin[1]:
                    curmin = item

        if classes[i] == classes[curmin[0]]:
            correct += 1.0
    return correct/distances.shape[0]


def FeatureSelection(data=None, debug=None):
    currentFeatureSet     = set()
    bestOverall           = set()
    globalBest: np.double = 0.0

    for i in range(1, data.shape[1]):
        bestSoFarAccuracy: np.double   = 0.0
        feature2AddAtCurrentLevel: int = -1
        for j in range(1, data.shape[1]):
            if j not in currentFeatureSet:
                tempFeatureSet = set(currentFeatureSet)
                tempFeatureSet.add(j)
                print(f'\t\tUsing Features: {tempFeatureSet}')
                accuracy: np.double = leave1OutCrossVal(curData=data[:, list(tempFeatureSet)], classes=data[:, 0])
                print(f'Accuracy is {round(accuracy*100, 3)}%')

                if accuracy > bestSoFarAccuracy:
                    bestSoFarAccuracy = accuracy
                    feature2AddAtCurrentLevel = j

        currentFeatureSet.add(feature2AddAtCurrentLevel)
        print(f'Feature Set: {currentFeatureSet} was best, accuracy is {round(bestSoFarAccuracy*100, 3)}%\n')

        if globalBest < bestSoFarAccuracy:
            globalBest  = bestSoFarAccuracy
            bestOverall = set(currentFeatureSet)
        else:
            print('(WARNING: Accuracy has decreased! Continuing search in case of local maxima)\n')

    print(f'\nFinished search! The best feature subset is {bestOverall}, yielding an accuracy of {round(globalBest*100, 3)}%.')
    return bestOverall


def FeatureBackwardSelection(data=None, debug=None):
    currentFeatureSet = set(range(1, data.shape[1]))
    bestOverall = set(range(1, data.shape[1]))
    print(f'\t\tUsing Features: {currentFeatureSet}')
    globalBest: np.double = leave1OutCrossVal(curData=data[:, list(currentFeatureSet)], classes=data[:, 0])
    print(f'Accuracy is {round(globalBest * 100, 3)}%')
    print(f'Feature Set: {currentFeatureSet} was best, accuracy is {round(globalBest * 100, 3)}%\n')
    for i in range(1, data.shape[1]):
        bestSoFarAccuracy: np.double = 0.0
        feature2RemoveAtCurrentLevel: int = -1

        for j in range(1, data.shape[1]):
            if j in currentFeatureSet:
                tempFeatureSet = set(currentFeatureSet) - {j}
                # print(f'\t\tUsing Features: {tempFeatureSet}')
                accuracy: np.double = leave1OutCrossVal(curData=data[:, list(tempFeatureSet)], classes=data[:, 0])
                print(f'Accuracy is {round(accuracy * 100, 3)}%')

                if accuracy > bestSoFarAccuracy:
                    bestSoFarAccuracy = accuracy
                    feature2RemoveAtCurrentLevel = j

        currentFeatureSet.remove(feature2RemoveAtCurrentLevel)
        print(f'Feature Set: {currentFeatureSet} was best, accuracy is {round(bestSoFarAccuracy * 100, 3)}%\n')

        if globalBest < bestSoFarAccuracy:
            globalBest = bestSoFarAccuracy
            bestOverall = set(currentFeatureSet)
        else:
            print('(Warning, Accuracy has decreased! Continuing search in case of local maxima)')

    print(f'\nFinished search! The best feature subset is {bestOverall}, yielding an accuracy of {round(globalBest * 100, 3)}%.')
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
    parser.add_option("--mode", dest='mode', default='1', help="1: Forward Search, 2: Backward Search, 3: Special Search on Language-accent Classification")
    (options, args) = parser.parse_args()
    try:
        main(options=options)
    except Exception as errorMain:
        print("Fail End Process: {0}".format(errorMain))
        traceback.print_exc()
    return


def main(options=None):
    print("Welcome to Rogelio\'s Feature Selection Algorithm.")
    if options.mode == '1' or options.mode == '2':
        if str.lower(options.filesize) == 'small':
            filename = 'CS205_small_testdata__37.txt'
        elif str.lower(options.filesize) == 'large':
            filename = 'CS205_large_testdata__4.txt'
        else:
            print(f'Invalid filesize chosen: {options.filesize}. Valid filesizes: [small | large]')
            return
        fileread = os.path.abspath(os.path.join(os.getcwd(), filename))
        print(f'File to test: {filename}')
        arr = np.loadtxt(fname=fileread)

    print('Algorithm options:')
    print('\t1) Forward Selection')
    print('\t2) Backward Selection')
    print('\t3) Forward Selection on Accent Classification')
    if options.mode == '1':
        bestFeatures = FeatureSelection(data=arr, debug=options.debug)
    elif options.mode == '2':
        bestFeatures = FeatureBackwardSelection(data=arr, debug=options.debug)
    elif options.mode == '3':
        # mapping = {'ES': 1, 'FR': 2, 'GE': 3, 'IT': 4, 'UK': 5, 'US': 6}
        mapping = {'ES': 2, 'FR': 2, 'GE': 2, 'IT': 2, 'UK': 2, 'US': 1}
        table = pd.read_csv('accent-mfcc-data.csv')
        for i in range(table.shape[0]):
            table.loc[i, 'language'] = mapping[table.iloc[i][0]]
        arr = table.values.astype(np.double)
        del table
        # bestFeatures = FeatureSelection(data=arr, debug=options.debug)
        bestFeatures = FeatureBackwardSelection(data=arr, debug=options.debug)
    else:
        print(f'Invalid mode chosen: {options.mode}. Valid modes: [1 | 2 | 3]')

    print(f'Mode option chosen: {options.mode}')
    return


if __name__ == "__main__":
    pStart = datetime.datetime.now()
    CLI()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop - pStart))


"""

Unused efficient data-processing code that still has potential for other dataset formats

"""
def filterWhiteSpace(elem):
    return elem != ''


def gen(iterableItem):
    for item in iterableItem:
        yield item


# lines = (line for line in open(fileread))
# linesAsLists = (s.rstrip().split(' ') for s in lines)
# lineSplits = (gen(filter(filterWhiteSpace, value)) for value in linesAsLists)
# lineFeatures = (value for value in lineSplits if value != '')
# for feats in lineFeatures:
#     for feat in feats:
#         pass
#         print(feat)
#         print(round(float(feat), 3))
