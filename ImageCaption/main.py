#UPDATE
#from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI,Request,Form,File,UploadFile
from fastapi.templating import Jinja2Templates

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle
import string
import tensorflow as tf
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
#from tensorflow.keras.utils import to_categorical, plot_model
#from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

#UPDATE
#BASE_DIR = r'C:\Users\hanum\Documents\Dataset\Flicker8k_Dataset'
BASE_DIR = r'C:\B20\Dataset\Flicker8k_Dataset'
#WORKING_DIR =
#UPDATE
#CAPTIONS_DIR = r'C:\Users\hanum\Documents\Dataset\Flicker8k_Text\Flickr8k.token.txt'
CAPTIONS_DIR = r'C:\B20\Dataset\Flicker8k_Text\Flickr8k.token.txt'


with open(os.path.join(CAPTIONS_DIR), 'r') as f:
    captions_doc = f.read()

mapping = {}
for line in (captions_doc.split('\n')):
    tokens = line.split('\t')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            caption = caption.translate(str.maketrans('','',string.punctuation))
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

pickle_features = pickle.load(open('features.pkl', 'rb'))

#UPDATE
#model_path = "model.h5"
model_path = "ImageCaptionModel.h5"
model = tf.keras.models.load_model(model_path)

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text

def generate_caption(image_name):
    image_id = image_name.split('.')[0]
#    img_path = os.path.join(BASE_DIR, image_name)
#    image = Image.open(img_path)
#    captions = mapping[image_id]
#    print('---------------------Actual---------------------')
#    for caption in captions:
#        print(caption)
    # predict the caption
    y_pred = predict_caption(model, pickle_features[image_id], tokenizer, max_length)
#    print('--------------------Predicted--------------------')
#    print(y_pred)
#    plt.imshow(image)
    return y_pred

#UPDATE
#@app.post("/predict")
#def caption(file: UploadFile = File(...)):
#    result = generate_caption(file.filename)
#    return {'Caption for the given image is': result}
#    #return {"filename": file.filename}

app = FastAPI()
templates = Jinja2Templates(directory = 'templates/')
@app.get('/')
def read_form():
    return 'hello world'

@app.get('/test2')
def read_form():
    return 'hello test2'

@app.get('/test')
def form_post(request:Request):
   res = '{{<Caption>}}'
   return templates.TemplateResponse('test.html',context  = {'request':request,'result':res})
   #return templates.TemplateResponse('test.html',context  = {'request':request,'result':res})