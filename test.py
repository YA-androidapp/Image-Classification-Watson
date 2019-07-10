#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

# $ python -m venv watson
# $ . watson/bin/activate
# (watson) $ pip install --upgrade 'ibm-watson'

from glob import glob
from ibm_watson import VisualRecognitionV3
import configparser
import datetime
import os
import json


scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにtestフォルダを作成、
# そのサブディレクトリに訓練データと検証データからなるデータセットを格納
root_test_dirname = 'test'

# テスト結果を出力するテキストファイル名
nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
result_filename = 'result-test-'+nowstr+'.txt'

# 正答率評価のための変数
count_items_all = 0
count_items_correct = 0

# 設定ファイル読み込み
inifile = configparser.ConfigParser()
inifile.read('./.key', 'UTF-8')
iam_api_key = inifile.get('watson', 'iam_api_key')

# Authentication
visual_recognition = VisualRecognitionV3(
    version='2019-06-22',
    iam_apikey=iam_api_key)

# Classify an image


def classify(path):
    try:
        with open(path, 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file,
                threshold='0.83',
                classifier_ids='DefaultCustomModel_1970291170').get_result()
        # print(json.dumps(classes, indent=2))
        # TODO: check
        # {
        #   "images": [
        #     {
        #       "classifiers": [
        #         {
        #           "classes": []
        label, score = classes['images'][0]['classifiers'][0]['classes'][0]['class'], classes['images'][0]['classifiers'][0]['classes'][0]['score']
        print(label, score)
        return label
    except Exception as e:
        print('[Err] {0}'.format(e))
    return ''


def main():
    global count_items_all
    global count_items_correct

    # テスト用画像取得
    subdirs = glob(os.path.join(
        scrpath, root_test_dirname, '**'))
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print('sub directory: {}'.format(subdir), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            testdatas = glob(os.path.join(subdir, '*.png'))
            count_testfile = 0
            answer = os.path.basename(subdir)
            for testdata in testdatas:
                print('  {} {:.2%} {}'.format(
                    answer, (count_testfile/len(testdatas)), testdata), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                label = classify(testdata)

                count_testfile += 1
                count_items_all += 1
                if label == answer:
                    count_items_correct += 1

    if count_items_all > 0:
        mes = 'Complete. accuracy:{} / {} = {:.2%}'.format(
            count_items_correct, count_items_all, count_items_correct / count_items_all)
    else:
        mes = 'Complete.'
    mes += ' ' + datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(mes)
    with open(os.path.join(scrpath, result_filename), mode='a') as f:
        f.write(mes + '\n')


if __name__ == '__main__':
    main()
