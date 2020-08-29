```python
from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features
```


```python
# 여기선 tensorflow 안쓴다. extracting features에 초점을 맞춘다.
#dataset path adn view passible targets
dataset_path = 'C:\\Users\\multicampus\\Desktop\\data_speech_commands_v0.02.tar\\data_speech_commands_v0.02'
# listdir => 현재 디렉토리에 있는 파일 리스트를 가져온다.
for name in listdir(dataset_path):
    # isdir => 디렉토리 경로가 존재하는지 체크
    # .join("/Users", "test") => 경로가 추가 된다. => /Users/test
    if isdir(join(dataset_path, name)):
        print(name)
```

    backward
    bed
    bird
    cat
    dog
    down
    eight
    five
    follow
    forward
    four
    go
    happy
    house
    learn
    left
    marvin
    nine
    no
    off
    on
    one
    right
    seven
    sheila
    six
    stop
    three
    tree
    two
    up
    visual
    wow
    yes
    zero
    _background_noise_
    


```python
# Create on all targets List
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)
print(dataset_path)
print(listdir(dataset_path))
print(join(dataset_path, name))
```

    ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero', '_background_noise_']
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02
    ['.DS_Store', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'LICENSE', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'README.md', 'right', 'seven', 'sheila', 'six', 'stop', 'testing_list.txt', 'three', 'tree', 'two', 'up', 'validation_list.txt', 'visual', 'wow', 'yes', 'zero', '_background_noise_']
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\_background_noise_
    


```python
# Leave off background noise set
all_targets.remove('_background_noise_')
print(all_targets)
```

    ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    


```python
# See how many files are in each
num_samples = 0
for target in all_targets:
    print(len(listdir(join(dataset_path, target))))
    num_samples += len(listdir(join(dataset_path, target)))
print('Total samples:', num_samples)
```

    1664
    2014
    2064
    2031
    2128
    3917
    3787
    4052
    1579
    1557
    3728
    3880
    2054
    2113
    1575
    3801
    2100
    3934
    3941
    3745
    3845
    3890
    3778
    3998
    2022
    3860
    3872
    3727
    1759
    3880
    3723
    1592
    2123
    4044
    4052
    Total samples: 105829
    


```python
### Settings
# all_targets: list에 러닝할 폴더 이름들이 들어가 있다.
target_list = all_targets
# npz 파일에 저장할 이미지
feature_sets_file = 'all_targets_mfcc_sets.npz'
# 기능 추출 시 오래걸리니 양을 줄인다.



# 임의의 데이터 하위집합 10% , 나중에 전체 개수에서 나눌거다.
perc_keep_samples = 1 # 1.0은 모든 samples이다



# 제대로 작동하는지 확인하는 것
# 교차 유효성 검사에 대한 데이터 10%
val_ratio = 0.1
# 테스트 데이터 10%
test_ratio = 0.1
# wav 파일이 16KHz sampling으로 기록되는 동안인 1분에 더빨리 기록되게한다
# 8KHz와 같이 낮은 sampling 속도로 악취 횟수를 설정
sample_rate = 8000
# 중격계수는 16
num_mfcc = 16
# MFCC 길이는 16
len_mfcc = 16
```


```python
# mfcc가 1분 동안 좋은 기능을 만들지 계속 생각해보자.
# Create List of filenames along with ground truth vector (y)
# 배열 만들기
filenames = []
y = []
for index, target in enumerate(target_list):
    print(join(dataset_path, target))
    # listdir => 현재 디렉토리에 있는 파일 리스트를 가져온다.
    filenames.append(listdir(join(dataset_path, target)))  # 단어당 안에 있는 음성파일을 []에 넣어서 추가 된다. 
    y.append(np.ones(len(filenames[index])) * index) # np.ones(len(filenames[index])) => [1]*음성파일 갯수.
    
```

    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\backward
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\bed
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\bird
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\cat
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\dog
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\down
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\eight
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\five
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\follow
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\forward
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\four
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\go
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\happy
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\house
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\learn
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\left
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\marvin
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\nine
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\no
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\off
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\on
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\one
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\right
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\seven
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\sheila
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\six
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\stop
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\three
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\tree
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\two
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\up
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\visual
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\wow
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\yes
    C:\Users\multicampus\Desktop\data_speech_commands_v0.02.tar\data_speech_commands_v0.02\zero
    


```python
# Check ground truth y vector
print(y)
for item in y:
    print(len(item))
```

    [array([0., 0., 0., ..., 0., 0., 0.]), array([1., 1., 1., ..., 1., 1., 1.]), array([2., 2., 2., ..., 2., 2., 2.]), array([3., 3., 3., ..., 3., 3., 3.]), array([4., 4., 4., ..., 4., 4., 4.]), array([5., 5., 5., ..., 5., 5., 5.]), array([6., 6., 6., ..., 6., 6., 6.]), array([7., 7., 7., ..., 7., 7., 7.]), array([8., 8., 8., ..., 8., 8., 8.]), array([9., 9., 9., ..., 9., 9., 9.]), array([10., 10., 10., ..., 10., 10., 10.]), array([11., 11., 11., ..., 11., 11., 11.]), array([12., 12., 12., ..., 12., 12., 12.]), array([13., 13., 13., ..., 13., 13., 13.]), array([14., 14., 14., ..., 14., 14., 14.]), array([15., 15., 15., ..., 15., 15., 15.]), array([16., 16., 16., ..., 16., 16., 16.]), array([17., 17., 17., ..., 17., 17., 17.]), array([18., 18., 18., ..., 18., 18., 18.]), array([19., 19., 19., ..., 19., 19., 19.]), array([20., 20., 20., ..., 20., 20., 20.]), array([21., 21., 21., ..., 21., 21., 21.]), array([22., 22., 22., ..., 22., 22., 22.]), array([23., 23., 23., ..., 23., 23., 23.]), array([24., 24., 24., ..., 24., 24., 24.]), array([25., 25., 25., ..., 25., 25., 25.]), array([26., 26., 26., ..., 26., 26., 26.]), array([27., 27., 27., ..., 27., 27., 27.]), array([28., 28., 28., ..., 28., 28., 28.]), array([29., 29., 29., ..., 29., 29., 29.]), array([30., 30., 30., ..., 30., 30., 30.]), array([31., 31., 31., ..., 31., 31., 31.]), array([32., 32., 32., ..., 32., 32., 32.]), array([33., 33., 33., ..., 33., 33., 33.]), array([34., 34., 34., ..., 34., 34., 34.])]
    1664
    2014
    2064
    2031
    2128
    3917
    3787
    4052
    1579
    1557
    3728
    3880
    2054
    2113
    1575
    3801
    2100
    3934
    3941
    3745
    3845
    3890
    3778
    3998
    2022
    3860
    3872
    3727
    1759
    3880
    3723
    1592
    2123
    4044
    4052
    


```python
# Flatten filename and y vectors
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]
```


```python
# Associate filenames with true output and shuffle
# >>> list(zip([1, 2, 3], [4, 5, 6]))   =>    [(1, 4), (2, 5), (3, 6)]
filenames_y = list(zip(filenames, y))
# shuffle은 리스트 항목 섞기
random.shuffle(filenames_y)
# 다시 unzip 한다 왜냐하면
filenames, y = zip(*filenames_y)
```


```python
# Only keep the specified number of samples (shorter extraction/training)
# 우리는 프로토 타입 모델 중 총 10%만 사용할 것이다.
# 여기서 중요한 것은 다시 돌아와서 모든 데이터를 사용하는 것이다.
print(len(filenames))
filenames = filenames[:int(len(filenames) * perc_keep_samples)]
print(len(filenames))
```

    105829
    105829
    


```python
# Calculate validation and test set sizes
# 모델을 교육 할 준비가 끝났다.
# 두 개의 개별 검증 테스트에서 파일 이름 목록 및 근거정보 목록을 wav 파일에서 function을 추출 할 준비가 됐다
val_set_size = int(len(filenames) * val_ratio)
test_set_size = int(len(filenames) * test_ratio)
```


```python
# Break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]
```


```python
# Break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]
```


```python
# Function: Create MFCC from given path
def calc_mfcc(path):
    
    #Load wavdfile
    # 초당 8000개의 샘플로 리샘플링 하는 Librosa를 사용하여 주어진 경로에서
    # wav파일을 빠르게 로드하자
    signal, fs = librosa.load(path, sr=sample_rate)
    
    # Create MFCCs from sound clip
    # MFCC 기능을 제공하는 python_speech_features를 사용하자
    # 기능들을 이용하여 해당 파형에서 MFCC 세트를 만들자 
    # 매개변수를 사용하여 MFCC set 수를 유지한다.
    # winlen은 25ms 에서 256ms 로 넓히자
    # winstep은 50ms 늘렸다
    # nFFT에 사용할 샘플 수는 window 크기에 따라 다르다.
    
    mfccs = python_speech_features.base.mfcc(signal,
                                            samplerate=fs,
                                            winlen=0.256,
                                            winstep=0.050,
                                            numcep=num_mfcc,
                                            nfilt=26,
                                            nfft=2048,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning)
    return mfccs.transpose()
```


```python
# signal - 특징을 계산할 오디오 신호, N*1 배열이어야 한다
# simplerate - 우리가 작업하고 있는 신호의 샘플링 속도
# winlen - 분석 창의 길이의 기본값은 0.025초이다
# winstep - 몇 초 안에 연속적인 윈도우 사이의 step. 기본값은 0.01초
# numcep - 반환되는 cepstrum 수, 기본값이 13
# nfilt - filterbank안의 filter의 수. 디폴트는 26
# nfft - FFT 사이즈. 디폴트는 512
# lowfreq - mel filters의 가장 낮은 band edge. 기본 Hz는 0
# highfreq - mel filters의 가장 높 band edge. 기본 Hz는 samplerate/2
# preemph - apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97
# ceplifter -apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22
# appendEnergy - if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
# returns - A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
```


```python
#위의 내용을 약간의 파일들에 테스트를 해보자 
# 처음에는 500개의 훈련 세트를 가져오자
# m 개의 FCC 매트릭스의 모양을 보자 각 오디오 파일을 16세트의 16세트를 생성해야한다.
# TEST: Construct test set by computing MFCC of each WAV file
prob_cnt = 0
x_test = []
y_test = []
for index, filename in enumerate(filenames_train):
    # Stop after 500
    if index >= 500:
        break
        
    # Create path from given filename and target item
    path = join(dataset_path, target_list[int(y_orig_train[index])],
               filename)
    
    # Create MFCCs
    mfccs = calc_mfcc(path)
    
    if mfccs.shape[1] == len_mfcc:
        x_test.append(mfccs)
        y_test.append(y_orig_train[index])
    else:
        print('Dropped:', index, mfccs.shape)
        prob_cnt += 1
```

    Dropped: 3 (16, 12)
    Dropped: 21 (16, 15)
    Dropped: 33 (16, 13)
    Dropped: 56 (16, 13)
    Dropped: 63 (16, 11)
    Dropped: 69 (16, 15)
    Dropped: 91 (16, 15)
    Dropped: 95 (16, 11)
    Dropped: 96 (16, 8)
    Dropped: 100 (16, 11)
    Dropped: 101 (16, 13)
    Dropped: 115 (16, 13)
    Dropped: 134 (16, 13)
    Dropped: 153 (16, 10)
    Dropped: 165 (16, 11)
    Dropped: 171 (16, 13)
    Dropped: 180 (16, 8)
    Dropped: 194 (16, 13)
    Dropped: 197 (16, 6)
    Dropped: 208 (16, 10)
    Dropped: 218 (16, 9)
    Dropped: 261 (16, 13)
    Dropped: 267 (16, 5)
    Dropped: 272 (16, 12)
    Dropped: 300 (16, 4)
    Dropped: 315 (16, 13)
    Dropped: 317 (16, 13)
    Dropped: 321 (16, 12)
    Dropped: 376 (16, 7)
    Dropped: 448 (16, 15)
    Dropped: 451 (16, 13)
    Dropped: 457 (16, 10)
    Dropped: 458 (16, 15)
    Dropped: 496 (16, 5)
    


```python
# 오디오 파일 중 손상(?) 된 것들의 갯수에 500을 나누면 1초도 안걸린다? 
print('% of problematic samples:', prob_cnt / 500)

```

    % of problematic samples: 0.068
    


```python
# 0.08이니 샘플의 손상간 부분은 약 10% 라는 결론이 나온다.
# 이중 몇개는 재생 사운드 라이브러리를 사용하여 단어를 여러개 테스트 하고 오디오 샘플의 MFCC와 결과 이미지를 보인다.
# 잘 들이거나 안들리는 것도 여러개 있다.
# TEST: Test shorter MFCC
# !pip install playsound
from playsound import playsound

idx = 46


# Create path from given filename and target item
path = join(dataset_path, target_list[int(y_orig_train[idx])],
           filenames_train[idx])

# Create MFCCs
mfccs = calc_mfcc(path)
print("MFCCs:", mfccs)

# Plot MFCC
fig = plt.figure()
plt.imshow(mfccs, cmap='inferno', origin='lower')

# TEST: play problem sounds
print(target_list[int(y_orig_train[idx])])
playsound(path)
############ 밑에 그림이 16x16 행렬 변환한거.
```

    MFCCs: [[-5.42517370e+01 -5.34845042e+01 -5.30167580e+01 -4.76347112e+01
      -3.30563455e+01 -2.17144299e+01 -1.71370585e+01 -1.65519251e+01
      -2.17639912e+01 -3.07394152e+01 -3.69535780e+01 -4.42623008e+01
      -5.03578020e+01 -5.57713068e+01 -5.76182060e+01 -5.80101134e+01]
     [ 3.57721794e+00  3.31342228e+00  4.80297664e+00  3.51345767e+00
       7.53533934e+00  1.07017412e+01  1.13372555e+01  1.10180747e+01
       1.06622674e+01  1.04938214e+01  1.06242197e+01  1.22014423e+01
       1.02742820e+01  9.14451618e+00  7.39796005e+00  7.39502410e+00]
     [ 2.06169606e+00  2.48659408e+00  3.03973638e+00  3.82446388e+00
      -2.40810563e+00 -5.38170872e+00 -5.57687876e+00 -2.46615843e+00
       3.81251497e+00  1.01736938e+01  9.42662646e+00  4.55204494e+00
       1.01650872e+00 -2.05100089e-01 -1.54943940e-01 -6.41865923e-02]
     [ 2.00906204e+00  2.22619070e+00  1.98788088e+00  2.88038004e+00
      -2.34130216e+00 -1.93705031e+00 -1.40692786e+00 -2.65440436e+00
      -4.21536997e+00 -4.16450057e+00 -1.42672539e+00  2.88229520e-01
      -8.63158824e-01 -3.41494757e-01  5.94722811e-01  6.46269841e-01]
     [-2.48744794e+00 -2.63033560e+00 -1.40284074e+00 -9.83368481e-02
      -4.90080882e-01 -1.47457202e+00 -2.97628202e+00 -3.99815534e+00
      -5.92546686e+00 -6.48655989e+00 -5.42151660e+00 -3.44097729e+00
      -2.47825229e+00 -2.12757692e+00 -1.44347062e+00 -9.05958657e-01]
     [ 8.57188085e-01  8.43488997e-01 -8.56962823e-02 -1.60453271e+00
      -3.23765213e-01  3.49143811e-01  8.66958545e-01 -2.36755229e-01
      -6.60545776e-01 -8.74128212e-01 -1.34157434e+00 -2.24839889e+00
      -2.42980159e+00 -2.24248713e+00 -1.67890479e+00 -1.25358938e+00]
     [-3.06886798e+00 -3.20714899e+00 -2.38601759e+00 -1.70329686e+00
       7.39755905e-02 -2.98825210e-01 -5.34571418e-01 -8.55156829e-01
      -1.53194410e+00 -3.81279841e+00 -4.52943863e+00 -3.68466200e+00
      -2.38881972e+00 -2.03949091e+00 -1.86884460e+00 -1.72120915e+00]
     [ 4.50454780e-01  4.28456596e-01 -2.68850330e-01 -7.46106551e-01
       4.82242475e-01  8.35613198e-01  1.03681842e+00  5.74725254e-01
       2.25492539e-01  8.91766425e-01  7.14178820e-01 -6.93446441e-01
      -1.01163240e+00 -3.41552064e-02  2.19626228e-01 -1.31952727e-01]
     [-9.50537838e-01 -1.21964478e+00 -1.66709688e+00 -2.28006503e+00
      -3.18219780e+00 -3.40055175e+00 -3.54542235e+00 -3.46441469e+00
      -3.24920383e+00 -1.78566217e+00 -9.57175326e-01 -4.57770929e-01
      -6.30514608e-01 -6.12502120e-01 -3.13795298e-01 -5.07793146e-01]
     [-1.57911782e-01 -3.79056559e-01 -2.46613751e-01 -8.69838277e-01
      -2.85636007e+00 -3.02053467e+00 -2.35637990e+00 -1.50636880e+00
      -8.73959943e-01 -1.22570939e+00 -1.20941126e+00 -8.94947963e-01
      -3.92351710e-01  2.88559045e-01  3.79013573e-01  1.85996770e-01]
     [ 1.17214824e-01  1.57058845e-01 -3.79645164e-01 -9.51461873e-01
      -7.89944081e-01 -2.49440424e-01 -5.27636379e-01 -7.21356426e-01
      -5.95178313e-01 -5.99533119e-01 -9.12481762e-01 -8.01504486e-01
      -1.01094873e+00 -1.11844802e+00 -1.05423089e+00 -8.69853642e-01]
     [-7.98570624e-01 -4.97405112e-01 -5.86010258e-01 -4.92034565e-01
      -2.22485289e-02 -9.69059218e-01 -9.53273548e-01 -5.69845043e-01
      -7.44071165e-01 -1.15991583e+00 -1.70042785e+00 -2.32171270e+00
      -1.57487037e+00 -1.10418830e+00 -1.34604156e+00 -1.17429311e+00]
     [-1.63727178e+00 -1.28992830e+00 -7.65096729e-01 -1.27259229e+00
      -1.54724791e+00 -1.63850862e+00 -1.77163700e+00 -2.16429713e+00
      -2.16656044e+00 -2.01060549e+00 -1.74367488e+00 -1.27129020e+00
      -8.37825208e-01 -9.54518604e-01 -1.11938475e+00 -1.07564840e+00]
     [ 2.53731335e-01  5.99048924e-01  2.53868095e-01  2.59253243e-01
      -7.74723482e-01 -1.29836633e+00 -1.62567560e+00 -1.60299628e+00
      -1.64955736e+00 -1.45135583e+00 -1.04226783e+00 -8.98903442e-01
       1.74812199e-01  3.56150086e-01  1.10042328e-01 -2.92642811e-01]
     [-9.31985996e-01 -5.36247854e-01 -3.18906299e-01  2.41145413e-01
      -7.10347425e-01 -1.45590312e+00 -1.54516343e+00 -1.65080936e+00
      -1.60428330e+00 -1.91885147e+00 -1.33765977e+00 -5.80949129e-01
       2.70049569e-01  3.25580689e-01 -2.34589124e-01 -7.81889546e-01]
     [-5.64506958e-01 -4.48838246e-01  2.61842333e-01  5.29511856e-01
      -3.88899711e-01 -9.96190883e-01 -1.38811914e+00 -1.33034775e+00
      -1.20200124e+00 -1.35265351e+00 -1.39501655e+00 -1.05242766e+00
      -3.52929185e-01 -2.57154147e-01 -6.81314688e-01 -9.32409857e-01]]
    five
    


![png](output_18_1.png)



```python
# 파일이 점으로 끝나는지 확인하는 함수
# 웨이브는 길이가 충분하지 않은 경우 Y벡터의 샘플 및 해당 레이블을 MFCC로 계산한다.
# function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []
    
    for index, filename in enumerate(in_files):
        
        # Create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])],
                   filename)
        
        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue
            
        # Create MFCCs
        mfccs = calc_mfcc(path)
        
        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1
            
    return out_x, out_y, prob_cnt
```


```python
# 교육 검증에서 테스트 세트를 활용해 해당 기능을 실행한다
# Create train, valudation, and test sets
x_train, y_train, prob = extract_features(filenames_train,
                                          y_orig_train)
print('Removed percentage:', prob / len(y_orig_train))
x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
print('Removed percentage:', prob / len(y_orig_val))
x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
print('Removed percentage:', prob / len(y_orig_test))
```

    Dropped: 3 (16, 12)
    Dropped: 21 (16, 15)
    Dropped: 33 (16, 13)
    Dropped: 56 (16, 13)
    Dropped: 63 (16, 11)
    Dropped: 69 (16, 15)
    Dropped: 91 (16, 15)
    Dropped: 95 (16, 11)
    Dropped: 96 (16, 8)
    Dropped: 100 (16, 11)
    Dropped: 101 (16, 13)
    Dropped: 115 (16, 13)
    Dropped: 134 (16, 13)
    Dropped: 153 (16, 10)
    Dropped: 165 (16, 11)
    Dropped: 171 (16, 13)
    Dropped: 180 (16, 8)
    Dropped: 194 (16, 13)
    Dropped: 197 (16, 6)
    Dropped: 208 (16, 10)
    Dropped: 218 (16, 9)
    Dropped: 261 (16, 13)
    Dropped: 267 (16, 5)
    Dropped: 272 (16, 12)
    Dropped: 300 (16, 4)
    Dropped: 315 (16, 13)
    Dropped: 317 (16, 13)
    Dropped: 321 (16, 12)
    Dropped: 376 (16, 7)
    Dropped: 448 (16, 15)
    Dropped: 451 (16, 13)
    Dropped: 457 (16, 10)
    Dropped: 458 (16, 15)
    Dropped: 496 (16, 5)
    Dropped: 533 (16, 7)
    Dropped: 552 (16, 12)
    Dropped: 600 (16, 11)
    Dropped: 637 (16, 11)
    Dropped: 660 (16, 11)
    Dropped: 667 (16, 14)
    Dropped: 692 (16, 11)
    Dropped: 701 (16, 13)
    Dropped: 726 (16, 14)
    Dropped: 730 (16, 11)
    Dropped: 754 (16, 12)
    Dropped: 756 (16, 15)
    Dropped: 757 (16, 14)
    Dropped: 780 (16, 15)
    Dropped: 787 (16, 15)
    Dropped: 797 (16, 6)
    Dropped: 803 (16, 12)
    Dropped: 816 (16, 11)
    Dropped: 817 (16, 11)
    Dropped: 819 (16, 15)
    Dropped: 837 (16, 13)
    Dropped: 847 (16, 8)
    Dropped: 850 (16, 13)
    Dropped: 864 (16, 11)
    Dropped: 865 (16, 13)
    Dropped: 900 (16, 10)
    Dropped: 925 (16, 14)
    Dropped: 928 (16, 14)
    Dropped: 947 (16, 8)
    Dropped: 950 (16, 15)
    Dropped: 951 (16, 13)
    Dropped: 976 (16, 11)
    Dropped: 1001 (16, 8)
    Dropped: 1015 (16, 9)
    Dropped: 1034 (16, 14)
    Dropped: 1049 (16, 13)
    Dropped: 1052 (16, 14)
    Dropped: 1061 (16, 15)
    Dropped: 1062 (16, 8)
    Dropped: 1066 (16, 10)
    Dropped: 1067 (16, 11)
    Dropped: 1081 (16, 14)
    Dropped: 1083 (16, 11)
    Dropped: 1102 (16, 12)
    Dropped: 1122 (16, 11)
    Dropped: 1125 (16, 14)
    Dropped: 1140 (16, 14)
    Dropped: 1146 (16, 13)
    Dropped: 1147 (16, 11)
    Dropped: 1151 (16, 11)
    Dropped: 1153 (16, 15)
    Dropped: 1174 (16, 11)
    Dropped: 1182 (16, 14)
    Dropped: 1187 (16, 15)
    Dropped: 1194 (16, 8)
    Dropped: 1205 (16, 14)
    Dropped: 1207 (16, 13)
    Dropped: 1271 (16, 9)
    Dropped: 1284 (16, 13)
    Dropped: 1286 (16, 11)
    Dropped: 1294 (16, 14)
    Dropped: 1301 (16, 10)
    Dropped: 1308 (16, 13)
    Dropped: 1309 (16, 15)
    Dropped: 1315 (16, 13)
    Dropped: 1329 (16, 11)
    Dropped: 1335 (16, 6)
    Dropped: 1352 (16, 13)
    Dropped: 1356 (16, 12)
    Dropped: 1360 (16, 14)
    Dropped: 1381 (16, 15)
    Dropped: 1389 (16, 13)
    Dropped: 1391 (16, 15)
    Dropped: 1394 (16, 12)
    Dropped: 1395 (16, 8)
    Dropped: 1424 (16, 8)
    Dropped: 1429 (16, 12)
    Dropped: 1432 (16, 11)
    Dropped: 1436 (16, 15)
    Dropped: 1447 (16, 10)
    Dropped: 1462 (16, 10)
    Dropped: 1478 (16, 15)
    Dropped: 1481 (16, 3)
    Dropped: 1499 (16, 10)
    Dropped: 1502 (16, 14)
    Dropped: 1518 (16, 15)
    Dropped: 1533 (16, 13)
    Dropped: 1545 (16, 13)
    Dropped: 1553 (16, 12)
    Dropped: 1568 (16, 13)
    Dropped: 1572 (16, 8)
    Dropped: 1582 (16, 14)
    Dropped: 1590 (16, 11)
    Dropped: 1592 (16, 14)
    Dropped: 1595 (16, 14)
    Dropped: 1600 (16, 13)
    Dropped: 1603 (16, 15)
    Dropped: 1605 (16, 14)
    Dropped: 1615 (16, 12)
    Dropped: 1619 (16, 10)
    Dropped: 1624 (16, 12)
    Dropped: 1629 (16, 12)
    Dropped: 1646 (16, 11)
    Dropped: 1649 (16, 13)
    Dropped: 1652 (16, 10)
    Dropped: 1654 (16, 12)
    Dropped: 1713 (16, 15)
    Dropped: 1720 (16, 15)
    Dropped: 1763 (16, 15)
    Dropped: 1776 (16, 8)
    Dropped: 1795 (16, 11)
    Dropped: 1796 (16, 9)
    Dropped: 1806 (16, 12)
    Dropped: 1816 (16, 12)
    Dropped: 1825 (16, 13)
    Dropped: 1839 (16, 8)
    Dropped: 1845 (16, 13)
    Dropped: 1863 (16, 15)
    Dropped: 1879 (16, 15)
    Dropped: 1880 (16, 14)
    Dropped: 1887 (16, 15)
    Dropped: 1909 (16, 13)
    Dropped: 1911 (16, 11)
    Dropped: 1929 (16, 14)
    Dropped: 1930 (16, 11)
    Dropped: 1947 (16, 8)
    Dropped: 1953 (16, 13)
    Dropped: 1963 (16, 13)
    Dropped: 1971 (16, 12)
    Dropped: 1976 (16, 8)
    Dropped: 1980 (16, 13)
    Dropped: 1994 (16, 11)
    Dropped: 1995 (16, 15)
    Dropped: 2017 (16, 4)
    Dropped: 2019 (16, 8)
    Dropped: 2034 (16, 11)
    Dropped: 2037 (16, 11)
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-22-7f6652938051> in <module>
          1 # 교육 검증에서 테스트 세트를 활용해 해당 기능을 실행한다
          2 # Create train, valudation, and test sets
    ----> 3 x_train, y_train, prob = extract_features(filenames_train,
          4                                           y_orig_train)
          5 print('Removed percentage:', prob / len(y_orig_train))
    

    <ipython-input-21-1dc3dfa3319e> in extract_features(in_files, in_y)
         18 
         19         # Create MFCCs
    ---> 20         mfccs = calc_mfcc(path)
         21 
         22         # Only keep MFCCs with given length
    

    <ipython-input-16-5d4948750a69> in calc_mfcc(path)
          5     # 초당 8000개의 샘플로 리샘플링 하는 Librosa를 사용하여 주어진 경로에서
          6     # wav파일을 빠르게 로드하자
    ----> 7     signal, fs = librosa.load(path, sr=sample_rate)
          8 
          9     # Create MFCCs from sound clip
    

    C:\ProgramData\Anaconda3\envs\tensorflow\lib\site-packages\librosa\core\audio.py in load(path, sr, mono, offset, duration, dtype, res_type)
        170 
        171     if sr is not None:
    --> 172         y = resample(y, sr_native, sr, res_type=res_type)
        173 
        174     else:
    

    C:\ProgramData\Anaconda3\envs\tensorflow\lib\site-packages\librosa\core\audio.py in resample(y, orig_sr, target_sr, res_type, fix, scale, **kwargs)
        582         y_hat = samplerate.resample(y.T, ratio, converter_type=res_type).T
        583     else:
    --> 584         y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)
        585 
        586     if fix:
    

    C:\ProgramData\Anaconda3\envs\tensorflow\lib\site-packages\resampy\core.py in resample(x, sr_orig, sr_new, axis, filter, **kwargs)
        118     x_2d = x.swapaxes(0, axis).reshape((x.shape[axis], -1))
        119     y_2d = y.swapaxes(0, axis).reshape((y.shape[axis], -1))
    --> 120     resample_f(x_2d, y_2d, sample_ratio, interp_win, interp_delta, precision)
        121 
        122     return y
    

    KeyboardInterrupt: 



```python
# 위의 결과는 simple중에 약 10프로 정도 제거 된 것을 알 수 있다.
# 마지막으로 numpy save Z 함수를 사용하여 이러한 대규모 배열을 NP에 저장한다.
np.savez(feature_sets_file,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test)
```


```python
# TEST: Load features
# numpy dot load라고 부르고 우리가 할 수 있는 파일의 위치를 알려준다.
# 사용 가능한 배열을 나열하고 각 배열의 샘플 수 를 확인하자
feature_sets = np.load(feature_sets_file)
feature_sets.files
```


```python
len(feature_sets['x_train'])
```


```python
# y 유효성 섬사 세트를 인쇄 하여 우리가 가진 모든 레이블을 볼 수 있다.
print(feature_sets['y_val'])
```


```python

```


```python

```
