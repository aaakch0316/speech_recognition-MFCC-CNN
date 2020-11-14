#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import lite
from tensorflow.keras import models


# In[2]:


# Parameters
keras_model_filename = 'wake_word_stop_model.h5'
tflite_filename = 'wake_word_stop_lite.tflite'


# In[3]:


# Convert model to TF Lite model
model = models.load_model(keras_model_filename)
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_filename, 'wb').write(tflite_model)


# In[ ]:




