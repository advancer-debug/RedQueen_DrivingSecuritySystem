#!/usr/bin/env python3
# coding=utf-8
'''
语音识别

利用pyaudio从麦克风采样获取PCM音频文件

百度语音识别SDK中要求：
＝＝>原始 PCM 的录音参数必须符合 8k/16k 采样率、16bit 位深、单声道，支持的格式有：pcm（不压缩）、wav（不压缩，pcm编码）、amr（压缩格式）。

PyAudio API：http://people.csail.mit.edu/hubert/pyaudio/docs/#pasampleformat
'''
_author_ = 'zixuwang'
_datetime_ = '2018-3-27'


import wave
from pyaudio import PyAudio,paInt16
import Voice2text

RATE=16000 #采样频率 8000 or 16000
CHUNK=2048
CHANNELS=1#声道
sampwidth=2#采样字节 1 or 2
TIME=3
SAVE_FILE = './Voices/save.wav'
SOUND_FILE = 'save.wav'
'''
保存音频文件
'''
def save_wave_file(filename,data):
    '''save the date to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sampwidth)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()


'''
开始录音
'''
def my_record():
    pa=PyAudio()
    stream=pa.open(format=paInt16,channels=1,
                   rate=RATE,input=True,
                   frames_per_buffer=CHUNK)
    # pyaudio.paInt16 = 8   ＝＝＝>    16 bit int
    #input – Specifies whether this is an input stream. Defaults to False.
    # frames_per_buffer – Specifies the number of frames per buffer.
    print('recording......')
    frames=[]
    #控制录音时间
    for i in range(0,int(RATE / CHUNK * TIME)):
        data = stream.read(CHUNK)
        frames.append(data)
        print('.')
    print('finish...')
    print()
    save_wave_file(SAVE_FILE,frames)
    stream.close()



def play():
    wf=wave.open(SOUND_FILE,'rb')
    p=PyAudio()
    stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=
    wf.getnchannels(),rate=wf.getframerate(),output=True)

    data = wf.readframes(CHUNK)
    while data != '' or data != " ":
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.start_stream()
    stream.close()
    p.terminate()


my_record()
Voice2text.voice2text()
# print('ok')
# play()