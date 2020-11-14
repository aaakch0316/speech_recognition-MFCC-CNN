import numpy as np
import scipy.signal
import timeit
import python_speech_features
import RPi.GPIO as GPIO

from tflite_runtime.interpreter import Interpreter


import pyautogui
import time


def ssafy_practice():
    pyautogui.click(71, 21)
    time.sleep(7)
    pyautogui.moveTo(487, 410, 2)
    pyautogui.click()
    time.sleep(7)
    pyautogui.moveTo(468, 277, 2)
    pyautogui.click()
    time.sleep(7)
    pyautogui.typewrite(['S','S','A','F','Y','enter'])


# Parameter
debug_time = 0
debug_acc = 1
led_pin = 8
# 문지방은 50%
word_threshold = 0.5
# recording duration - 0.5초
rec_duration = 0.5
# 0.5 초
window_stride = 0.5
# 마이크는 적어도 44000Hz의 샘플링 비율이 필요하다 
# sample_rate와 resample_rate는 아래의 decimate의 old_fs와 new_fs에 들어간다.
sample_rate = 48000
resample_rate = 8000
# 하나의 채널을 유지한다.
num_channels = 1
# 훈련에 16 MFCCC 사용
num_mfcc = 16
# tensorflow lite model은 샘플 오디오 데이터를 위한 buffer을 작동시키는 nunpy array을 생성한다.
model_path = 'wake_word_stop_lite.tflite'

# Sliding window
# np.zeros(5) // array([ 0.,  0.,  0.,  0.,  0.])
# np.zeros((5,), dtype=int) // array([0, 0, 0, 0, 0])
window = np.zeros(int(rec_duration * resample_rate) * 2)

#############################################################
# GPIO 현재 사용 안하는 코드. 혹시 몰라서 남겨둠
# 해석 객체를 만들고 TFlight에 경로를 준다.
# 그후에 모델로 부터 tensor에 할당하는 것이 필요하다
# GPIO.setwarnings(False)는 코드를 실행했을 때 setwarning false 오류가 뜨는 사람만 적어주면 된다.
GPIO.setwarnings(False)
# GPIO.setmode는 두가지 모드가 존재하는데 BOARD와 BCM이다.
# BOARD는 라즈베리파이 보드 번호로 사용하는 모드이고, (아래 GPIO.setup에 8이 적혀있다.)
# BCM(Broadcom chip-specific pin numbers)은 핀의 번호, 즉 gpio 21번 이라고 하면 21번이라는 번호를 사용하겠다는 뜻이다.
GPIO.setmode(GPIO.BOARD)
# GPIO.OUT-> 사용할 핀 출력 설정 / GPIO.LOW 사용할  led 초기화(불꺼짐)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)
#############################################################



# Load the TFLite model and allocate tensors.
# 만약 인풋과 아웃푹의 shape를 보기위한다면, interpreter로부터 pint함으로서 이러한 세부사항을 얻을 수 있다.
interpreter = Interpreter(model_path)
# 텐서에 할당하기
interpreter.allocate_tensors()
# Get input and output tensors.
# 입력 텐서 정보 : 인덱스 알아야 데이터 전달 가능
# [{'name': 'conv2d_60_input', 'index': 3, 'shape': array([  1, 150, 150,   3], dtype=int32),
input_details = interpreter.get_input_details()
# 출력 텐서 정보 : 인덱스를 알아야 결과를 가져올 수 있음.
# [{'name': 'dense_41/Sigmoid', 'index': 18, 'shape': array([1, 1], dtype=int32),
output_details = interpreter.get_output_details()
print(input_details)

# aliasing -위신호 현상은 신호 처리에서 표본화를 하는 가운데 각기 다른 신호를 구별해내지 못하게 하는 효과를 가리킨다. 신호가 샘플로부터 다시 구성될 때 결과가 원래의 연속적인 신호와 달라지는 "일그러짐"을 가리킨다. 계단 현상으로 부르기도 한다.
# aliasing을 줄이기 위해 sample을 줄이는 것이 필요하다.

# Decimate (filter and downsample)
# Decimation - 데시메이션은 사전에 찾아보면 대량살해, 대량파괴로 나와 있다. 신호처리에서 사용하는 의미는 혼신없는 대역폭축소를 의미한다. 디지털 신호처리에서, decimation은 신호의 샘플링 레이트를 줄이는 과정이다.
# 필터링과 다운 샘플링 과정은 decimation으로 알려져있다. 
# 그래서 우리는 scipy의 decimate function을 사용한다.
# 주목하자. 요소틀을 통함함으로서 신호들을 decimate 할 수 있다.
# 이를 수행하는 함수는 신호를 정수로만 제거 가능하다. === int(dec_factor)
# 위의 이유 때문에 우리는 48KHz의 sample로 나눌 수 있는 값이 처음에 필요하다. 왜냐하면 8000으로 나눈다.
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate highter than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    # 1.5 => is_integer => False
    # 1.0 => is_integer => True
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs
    
    # Do decimation
    # decimate(x, r)은 입력 신호 x의 sample 비율을 r배 만큼 줄인다.(=다운 샘플링)
    # 파동의 진폭이 줄어드는게 아니고 데이터들의 수를 줄여 파형을 유지하면서 간격을 늘린다.
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))
    # signal : N 차원 배열(Ndarray)로 다운 샘플링 할 signal이다
    # int(dec_factor)은 정수여야하고 다운 샘플링의 인자이다.

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
    
    GPIO.output(led_pin, GPIO.LOW)
    
    # Start timing for testing
    # timeit는 해당 프로그램이 돌아간 시작을 측정하는 거다.
    # 다음 timeit.default_timer()까지의 시간 초가를 알 수 있다.
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    # squeeze는 차원을 축소를 담당한다
    # np.sqeeze(배열, 축) -을 통해 지정된 축의 차원을 축소할 수 있다. 단, 축을 입력하지 않으면 1차원 배열로 축소한다.
    # [[1], [2], [3]] => squeeze => [1 2 3]
    rec = np.squeeze(rec)
    
    # Resample
    # 출력된 rec => 입력 신호 x의 sample 비율을 줄인 결과값
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec
    
    # Compute features
    mfccs = python_speech_features.base.mfcc(window,
    # window – the audio signal from which to compute features. Should be an N*1 array
                                             samplerate=new_fs,
    # samplerate – the samplerate of the signal we are working with.
                                             winlen=0.256,
    # winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
                                             winstep=0.050,
    # winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
                                             numcep=num_mfcc,
    # numcep – the number of cepstrum to return, default 13
                                             nfilt=26,
    # nfilt – the number of filters in the filterbank, default 26.
                                             nfft=2048,
    # nfft – the FFT size. Default is 512.
                                             preemph=0.0,
    # preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
                                             ceplifter=0,
    # ceplifter – apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
                                             appendEnergy=False,
    # appendEnergy – if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
                                             winfunc=np.hanning)
    # winfunc – the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming

    #python_speech_features.base.mfcc의 리턴 값
    # => A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.

    # 위 코드의 transpose 함수는 다차원 텐서을 변형(Transpose)하는 작업을 수행합니다(직관적 이해는 힘들다.)
    # 위에서 변형이란 전치를 의미한다.
    # 행렬의 전치란 지정한 축으로 텐서의 Shape을 변경하고, 새로운 Shape 구조에 맞도록 각 요소의 위치를 변경하는 과정이다.
    # numpy의 transpose 함수는 n 차원 텐서를 전치시키는 기능을 제공합니다. 이 함수는 변환할 텐서와 변환 기준 축(axis)을 입력 매개변수로 갖습니다. axis는 shape과 인덱스의 순서를 숫자로 나타내는 튜플입니다.
    # numpy의 transpose 함수는 axis 튜플을 기준으로 텐서의 shape을 변경하고, axis 튜플을 기준으로 요소의 인덱스를 변경하고 재배치합니다.
    mfccs = mfccs.transpose()
####### 직관적 이해
# ex
# > print(a)
# < array([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14]])
# > print(np.transpose(a))
# < array([[ 0,  5, 10],
#         [ 1,  6, 11],
#         [ 2,  7, 12],
#         [ 3,  8, 13],
#         [ 4,  9, 14]])



    # Make prediction from model
    # np.reshape(변경할 배열, 차원) / 배열.reshape(차원)
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    # set_tensor(tensor_index, value) => input tensor의 값을 정한다
    # tensor_index => 셋팅할 텐서의 인덱스 값, 인풋 디테일의 인덱스 필드에서 가져온다.
    # tensor의 셋팅 값이다.
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    # Run inference by invoking the `Interpreter`.
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('#######################')
    print(output_details)
    print(output_details[0]['index'])
    val = output_data[0][0]
    print(output_data)
    print(val)
    if val > word_threshold:
        print('stop')

        # 아래 부분은 gpio 부분.
        # GPIO.output(led_pin, GPIO.HIGH)
    
    if debug_acc:
        print(val)
        
    if debug_time:
        print(timeit.default_timer() - start)
        
# Start streaming form microphon
with sd.InputStream(channels=num_channels,
                     samplerate=sample_rate,
                     blocksize=int(sample_rate * rec_duration),
                     callback=sd_callback):
    
    while True:
        pass