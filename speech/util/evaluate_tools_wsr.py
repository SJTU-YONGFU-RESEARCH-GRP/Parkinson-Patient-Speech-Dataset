#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from multiprocessing.dummy import Pool

from attrdict import AttrDict

from util.text import levenshtein

import os
import Levenshtein
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def word_levenshtein_editops_list(source, target):
    unique_elements = sorted(set(source + target)) 
    char_list = [chr(i) for i in range(len(unique_elements))]
    if len(unique_elements) > len(char_list):
        raise Exception("too many elements")
    else:
        unique_element_map = {ele:char_list[i]  for i, ele in enumerate(unique_elements)}
    source_str = ''.join([unique_element_map[ele] for ele in source])
    target_str = ''.join([unique_element_map[ele] for ele in target])
    transform_list = Levenshtein.editops(source_str, target_str)
    
    return transform_list

def word_get_operation_counts(source, target): 
    transform_list = word_levenshtein_editops_list(source, target)
    substitutions = sum(1 if op[0] == "replace" else 0 for op in transform_list)
    deletions = sum(1 if op[0] == "delete" else 0 for op in transform_list)
    insertions = sum(1 if op[0] == "insert" else 0 for op in transform_list)
    hits = len(source) - (substitutions + deletions)

    return hits, substitutions, deletions, insertions

def pmap(fun, iterable):
    pool = Pool()
    results = pool.map(fun, iterable)
    pool.close()
    return results


def wer_cer_batch(samples):
    r"""
    The WER is defined as the edit/Levenshtein distance on word level divided by
    the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    wer = sum(s.word_distance for s in samples) / sum(s.word_length for s in samples)
    cer = sum(s.char_distance for s in samples) / sum(s.char_length for s in samples)

    wsr = sum(s.substitutions for s in samples) / sum(s.word_length for s in samples)
    wdr = sum(s.deletions for s in samples) / sum(s.word_length for s in samples)
    wir = sum(s.insertions for s in samples) / sum(s.word_length for s in samples)

    wer = min(wer, 1.0)
    cer = min(cer, 1.0)
    wsr = min(wsr, 1.0)
    wdr = min(wdr, 1.0)
    wir = min(wir, 1.0)

    return wer, cer, wsr, wdr, wir


def process_decode_result(item):
    ground_truth, prediction, loss = item
    char_distance = levenshtein(ground_truth, prediction)
    char_length = len(ground_truth)
    word_distance = levenshtein(ground_truth.split(), prediction.split())
    word_length = len(ground_truth.split())
    wer = word_distance / word_length
    cer = char_distance / char_length
    wer = min(wer, 1.0)
    cer = min(cer, 1.0)

    hits, substitutions, deletions, insertions = word_get_operation_counts(ground_truth.split(), prediction.split())
    wsr = substitutions / word_length
    wdr = deletions / word_length
    wir = insertions / word_length

    return AttrDict({
        'src': ground_truth,
        'res': prediction,
        'loss': loss,
        'char_distance': char_distance,
        'char_length': char_length,
        'word_distance': word_distance,
        'word_length': word_length,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'cer': cer,
        'wer': wer,
        'wsr': wsr,
        'wdr': wdr,
        'wir': wir,
    })


def calculate_report(labels, decodings, losses):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = pmap(process_decode_result, zip(labels, decodings, losses))

    # Getting the WER and CER from the accumulated edit distances and lengths
    samples_wer, samples_cer, samples_wsr, samples_wdr, samples_wir = wer_cer_batch(samples)

    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)

    # Then order by WER (highest WER on top)
    samples.sort(key=lambda s: s.wer, reverse=True)

    return samples_wer, samples_cer, samples_wsr, samples_wdr, samples_wir, samples

def convert_cm(source, target):
    editops = Levenshtein.editops(source, target)

    actual = list(source)
    predicted = list(source)
    #print(editops[0][1])
    for op in editops:
        if op[0] == "delete":
            m = op[1]
            #print(before,',',after)
            predicted[m] = 'D'
            #print(after)
        if op[0] == "replace":
            m = op[1]
            n = op[2]
            predicted[m] = target[n]

    cm = confusion_matrix(actual, predicted, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", ' ', 'D'])
    #cm = confusion_matrix(source, target)
    return cm

def get_cm(labels, decodings, losses, cm_dir):
    #print(source)
    matrix = np.zeros((29,29))
    samples = pmap(process_decode_result, zip(labels, decodings, losses))
    for s in samples:
        matrix += convert_cm(s.src, s.res)

    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)
    filename = cm_dir + '/' + 'cm1.csv'
    np.savetxt(filename, matrix, fmt='%d')
