#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
from os.path import isdir, join
from tensorflow.keras import layers, models
import numpy as np


# In[ ]:





# In[2]:


# Create List of all targets (minus background noise)
dataset_path = 'C:\\Users\\multicampus\Desktop\\data_speech_commands_v0.02.tar\\data_speech_commands_v0.02'
all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
all_targets.remove('_background_noise_')
print(all_targets)


# In[3]:


# step 1에서 만든 기능 세트 파일의 경로 설정
# Settings
feature_sets_path = 'C:\\Users\\multicampus\\Desktop\pjt_2\\s03p13d204\embedded\\tensorflow\\audio_recognition'
# 신경망 모델 저장 파일
feature_sets_filename = 'all_targets_mfcc_sets.npz'
# 파일이름을 지정하고 weights를 정의 할 것이다.

model_filename = 'wake_word_stop_model.h5'
# 위를 보고 사용가능한 단어여야 한다.

wake_word = 'stop'


# In[4]:


# 데이터 세트의 대상 numpy를 사용하여 기능 세트 파일을 로드한다.
# Load feature sets
feature_sets = np.load(join(feature_sets_path, feature_sets_filename))
print(feature_sets.files)
print(len(feature_sets.files[0]))


# In[5]:


# 텐서는 여러개로 정렬된 데이터 모음이다
# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']


# In[6]:


# 행렬의 크기는 단순히 2차원 텐서
# 즉 MFCC로 생성한 것은 단순한 행렬
# 첫번째 - 해당 세트의 셈플 수
# 나머지 2개는 계수의 수와 집합의 수이다.
# Look at tensor dimensions
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)


# In[7]:


# peek at labels
print(y_val)


# In[8]:


print(len(all_targets))


# In[9]:


# Convert ground truth arrays to one wake word (1) and 'other' (0)
# 0,1로 바꿔야 한다.
wake_word_index = all_targets.index(wake_word)
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')


# In[10]:


# Peek at Labels after conversion
print(y_val)


# In[11]:


# what percentage of 'stop' appear in validation Labels
print(sum(y_val) / len(y_val))
print(1 - sum(y_val) / len(y_val))


# In[12]:


# View the dimensions of our input data
print(x_train.shape)


# In[13]:


# CNN for TF expects (batch, height, width, channels)
# so we reshape the input tensors with a "color" channel of 1
x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_val = x_val.reshape(x_val.shape[0],
                      x_val.shape[1],
                      x_val.shape[2],
                      1)
x_test = x_test.reshape(x_test.shape[0],
                       x_test.shape[1],
                       x_test.shape[2],
                       1)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)


# In[14]:


# 요소가 하나뿐이기 때문에 배열이 하나다.
# Input shape for CNN is size of MFCC of 1 sample
sample_shape = x_test.shape[1:]
print(sample_shape)


# In[15]:


# Build model
# Based on: https://www.geeksforgeeks.org/python-image-classification-using-keras/

model = models.Sequential()
model.add(layers.Conv2D(32,
                       (2, 2),
                       activation='relu',
                       input_shape=sample_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2))) 
  
model.add(layers.Conv2D(32, (2, 2), activation='relu')) 
model.add(layers.MaxPooling2D(pool_size=(2, 2))) 
  
model.add(layers.Conv2D(64, (2, 2), activation='relu')) 
model.add(layers.MaxPooling2D(pool_size=(2, 2))) 
  
# Classifier
model.add(layers.Flatten()) 
model.add(layers.Dense(64)) 
model.add(layers.Activation('relu')) 
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(1, activation='sigmoid')) 


# In[16]:


# 위의 출력은 다음과 함께 완전히 연결되거나 조밀한 신경망에 공급
# display model
model.summary()


# In[17]:


# Add training parameters to model
# MFCC와 모양과 크기가 같은 다른 것들과 함께 세트로 구성
model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['acc'])


# In[18]:


# hyperparameters - 자동으로 업데이트 되지 않음
# 모델 설정 했으니 학습해야한다.
# loss 값 설정
model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['acc'])


# In[19]:


# fit 함수 호출하여 수행하고 훈련을 전달한다.
# Train
history = model.fit(x_train,
                   y_train,
                   epochs=40,
                   batch_size=1000,
                   validation_data=(x_val, y_val))

# 실측 답변, 샘플, 통과한 epochs 수, 배치 크기
# epochs와 batch_size를 가지고 놀면서 정확도를 높일 수 있다.
# 보이지 않는 데이터를 예측시 얼마나 잘 수행되는지를 확인


# In[20]:


# 마지막 부분을 확인하면 val_loss는 모델이 얼마나 잘 수행되었는지 알려주는 정확도 점수
# val_acc는 모델이 얼마나 잘 수행되었는지 알려주는 검증 정확도 점수
# 교차 검증하기 위해 정확도와 손실을 함수ㄹ확도가 주어진 두 개의 플롯을 만들자.
# Plot results
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[21]:


# 몇 번의 에포크 이후에도 커브가 여전히 심하게 흔들리면 훈련중 문제가 발생한거다.


# In[22]:


#save the model as a fil
models.save_model(model, model_filename)


# In[23]:


# 새로운 데이터에 대한 예측을 하려면, 먼저 따로 설정한 테스트데이터를 사용하여테스트 할 수 있다.
# 아래는 단어를 가리키는 훈련 데이터에 대한 모든 색인을 인쇄한다.
for idx, y in enumerate(y_test):
    if y == 1:
        print(idx)


# In[24]:


# 파일에서 모델을 로드하고 10에서 출력을 예측한다. 0.5 이상이 우리의 것.
# TEST: Load model and run it against test set
model = models.load_model(model_filename)
for i in range(0, 200):
    print('Answer:', y_test[i], 'Prediction:', model.predict(np.expand_dims(x_test[i], 0)))


# In[25]:


model.evaluate(x=x_test, y=y_test)


# In[26]:


# 위의 결과가 좀 더 검증 정확도에 가깝다. 우리의 모델은 1초의 오디오 클립을 예측할 때 약 98.4% 정확도이다.


# In[ ]:




