#!/usr/bin/env python
# -*- coding: utf-8 -*-

# $ python -m venv watson
# $ . watson/bin/activate
# (watson) $ pip install --upgrade 'ibm-watson'

from ibm_watson import VisualRecognitionV3
import configparser
import json

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
        label, score = classes['images'][0]['classifiers'][0]['classes'][0][
            'class'], classes['images'][0]['classifiers'][0]['classes'][0]['score']
        print(label, score)

    except Exception as e:
        print('[Err] {0}'.format(e))


def main():
    classify('test.png')


if __name__ == '__main__':
    main()
