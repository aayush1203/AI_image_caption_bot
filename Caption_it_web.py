#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.layers.merge import add


# In[4]:


model = load_model("./Model_weights/model_9.h5")
model._make_predict_function()

# In[5]:


model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[6]:


# create a new model, by removing the last layer (output layer of 1000 classes) from the resnet50
model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
model_resnet._make_predict_function()

# In[33]:


with open("./storage/word_to_idx.pkl",'rb') as w2i:
    word_to_idx = pickle.load(w2i)

with open("./storage/idx_to_word.pkl",'rb') as i2w:
    idx_to_word = pickle.load(i2w)
    


# In[34]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    
    img = img.reshape((1,224,224,3))
    
    # normalization (using resnet50 function)
    img = preprocess_input(img)
    
    return img

def encode_img(img):
    # preprocess the image
    img = preprocess_img(img)
    
    # pass image to resnet model
    feature_vector = model_resnet.predict(img)
    # print(feature_vector.shape)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    #print(feature_vector.shape)
    return feature_vector


# In[35]:


max_len=35
def predict_caption(photo):
    
    input_text = "startseq"
    
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in input_text.split() if w in word_to_idx ]
        # add padding
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        y_pred = model.predict([photo,sequence])
        y_pred = y_pred.argmax() # word with max sampling - Greedy sampling
        word = idx_to_word[y_pred]
        input_text += ' ' + word
        
        if word == 'endseq':
            break
    
    final_caption = input_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[36]:

def caption_this_image(image):
    enc = encode_img(image)
    caption = predict_caption(enc)
    
    return caption

# path = "./static/DSC_7802.JPG" 
# print(path)
# caption = caption_this_image(path)
# print(caption)