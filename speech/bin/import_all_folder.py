#!/usr/bin/env python

'''
    NAME    : LDC TIMIT Dataset
    URL     : https://catalog.ldc.upenn.edu/ldc93s1
    HOURS   : 5
    TYPE    : Read - English
    AUTHORS : Garofolo, John, et al.
    TYPE    : LDC Membership
    LICENCE : LDC User Agreement
'''

import errno
import os
from os import path
import sys
import tarfile
import fnmatch
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split

def clean(word):
    # LC ALL & strip punctuation which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new

def _preprocess_data(filepath):

    # Assume data is downloaded from LDC - https://catalog.ldc.upenn.edu/ldc93s1

    target = filepath
    print("Building CSVs")

    # Lists to build CSV files
    all_list_wavs, all_list_trans, all_list_size = [], [], []

    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.wav"):
            full_wav = os.path.join(root, filename)
            wav_filesize = path.getsize(full_wav)

            # need to remove _rif.wav (8chars) then add .TXT
            trans_file = full_wav[:-4] + ".txt"
            with open(trans_file, "r") as f:
                for line in f:
                    split = line.split()
                    #start = split[0]
                    #end = split[1]
                    t_list = split[0:]
                    trans = ""

                    for t in t_list:
                        trans = trans + clean(t) + " "
            all_list_wavs.append(full_wav)
            all_list_trans.append(trans)
            all_list_size.append(wav_filesize)

    all = {'wav_filename': all_list_wavs,
           'wav_filesize': all_list_size,
           'transcript': all_list_trans}

    df_all = pd.DataFrame(all, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)

    return df_all

def split(data, ratio1, ratio2):
    train_split_ratio = ratio1 # 60%的训练数据
    dev_split_ratio = ratio2
    seed = 5  # 随机种子
    
    # 分割训练集与测试集
    train, other = train_test_split(data, train_size=train_split_ratio, test_size=1-train_split_ratio, random_state=seed)
    dev, test = train_test_split(other, train_size=dev_split_ratio, test_size=1-dev_split_ratio, random_state=seed)
    
    return train, dev, test

if __name__ == "__main__":
    filepath = sys.argv[1]
    #filename = os.path.split(filepath)[1]
    #df_all = _preprocess_data(filepath)

    #dir_name = filepath + "/" + filename + "_all.csv"
    #df_all.to_csv(dir_name, sep=',', header=True, index=False, encoding='ascii')

    train = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    dev = pd.DataFrame(columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    test = pd.DataFrame(columns = ['wav_filename', 'wav_filesize', 'transcript'], dtype=int)

    if sys.argv[2]:
        ratio1 = float(sys.argv[2])
        ratio2 = float(sys.argv[3])
        
        for root, dirnames, filenames in os.walk(filepath):
            for dirname in dirnames:
                subfile = os.path.join(root, dirname)
                df_all = _preprocess_data(subfile)
                df_train, df_dev, df_test = split(df_all, ratio1, ratio2)
                train = train.append(df_train)
                dev = dev.append(df_dev)
                test = test.append(df_test)

        test_filename = filepath + '/' + os.path.split(filepath)[1] + '_test'  +'.csv'
        dev_filename = filepath + '/' + os.path.split(filepath)[1] + '_dev' +'.csv'
        train_filename = filepath + '/' + os.path.split(filepath)[1] + '_train' +'.csv'

        test.to_csv(test_filename, index=False, encoding='utf-8')
        dev.to_csv(dev_filename, index=False, encoding='utf-8')
        train.to_csv(train_filename, index=False, encoding='utf-8')
    
    print("Completed")
