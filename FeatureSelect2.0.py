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

Usage: $: python3 trenchPuzzleSearch.py -i 0234567891 -r 357
Args:
    --mode: ('small' | 'large')
    --debug: stores True
"""


import json
import optparse
import os
import datetime
import traceback
import numpy as np


def FeatureSelection(data=None, mode=None, debug=None):

    return []

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
            # print(feat)
            # print(round(float(feat), 3))
    arr = np.loadtxt(fname=fileread)
    print(arr.shape)
    bestFeatures = FeatureSelection(data=arr, mode=options.mode, debug=options.debug)
    return



if __name__ == "__main__":
    pStart = datetime.datetime.now()
    CLI()
    qStop = datetime.datetime.now()
    print("Execution time: " + str(qStop - pStart))
