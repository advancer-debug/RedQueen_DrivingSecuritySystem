#!/usr/bin/env python3
# coding=utf-8
'''
语音识别

从.wav语音文件中获取音频文件，转换成文字保存返回
远程调用百度AI的接口，需要联网使用
SDK：http://ai.baidu.com/docs#/ASR-Online-Python-SDK/top
'''
_author_ = 'zixuwang && xiameng'
_datetime_ = '2018-3-26'
from aip import AipSpeech
import os
import sys


APP_ID = ''
API_KEY = ''
SECRET_KEY = ''
SOURCE_PATH = './Voices/save.wav'

# 远程登录百度AI－语音识别
def get_api_object():
    return AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def append_to_file(results):
    with open('./Text/text.txt', 'a', encoding='UTF-8') as f:
        for r in results:
            f.write(r)
            f.write('\r\n')
            f.close()


# 将.wav波音音频文件中的音频转换成文字
def voice2text(sound_file='./Voices/save.wav'):
    api_object = get_api_object()
    file = open(sound_file, 'rb')
    result = api_object.asr(file.read(), 'wav', 16000, {'dev_pid':'1537',} )
    # 格式为wav，采样频率为16k，1537代表可以有标点的普通话
    print(result.get('result', ['Nothing']))
    # if(result.get('err_no') == 0):
        # append_to_file(result.get('result'))

# voice2text(SOURCE_PATH)
