import speech_recognition as speechrec
# import librosa # to extract speech features
import librosa.display
import soundfile # to read audio file
import scipy.io.wavfile as wav
from speechpy.feature import mfcc

from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd

import math
import re
import glob
import os
import sys
from tqdm import tqdm
from typing import Tuple

import sklearn
from sklearn.model_selection import train_test_split # for splitting training and testing
# from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC, LinearSVC
# from keras import Sequential
# from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
    # BatchNormalization, Activation, MaxPooling2D
from sklearn.metrics import accuracy_score, confusion_matrix
# from keras.models import load_model

import pickle # to save model after training
# import tensorflow as tf


MODEL_STORAGE_ML = "./models/" #Stores all the saved pickle files for all the models
TRANSCRIPTION_PATH = './transcriptions/'
WAV_FILE_PATH = './wav/' #directory path containing all test wav files
DATA_DIR = './pre-processed/' #directory to store all the preprocessed audio vectors
LSTM_PATH ='./models/20220301_071644/' #lstm saved model in google drive
CNN_2_PATH = './models/20220301_075730/' #cnn saved model in goofle drive (abstract class)
# NLP_CNN_PATH = 'gdrive/MyDrive/Major Project Sem 7 8/NLP/CNN/'
max_seq_len = 500
NLP_WEIGHTAGE = 0.4
SER_WEIGHTAGE = 0.6

filereader = open('assets/buzzword.txt','r')
buzzwords = filereader.read().splitlines()

sr = 44100
audio_vectors = {}
text_df = pd.DataFrame(columns=['wav_file','text', 'text_emotion_hap', 'text_emotion_ang', 'text_emotion_sad', 'text_emotion_neu','buzzword_present'])

df_testaudios= pd.DataFrame(columns=['wav_file','emotion']) #Dataframe to store the names of all the wav files and corresponding predicted emotion

orig_wav_filenames = []

orig_wav_files = os.listdir(WAV_FILE_PATH) #lists all the wav files in the directory

"""Creating all the audio vectors from the wav files using librosa library"""

for orig_wav_file in tqdm(orig_wav_files):
    try:
        orig_wav_vector, _sr = librosa.load(WAV_FILE_PATH + orig_wav_file, sr=sr)
        orig_wav_file, file_format = orig_wav_file.split('.')
        truncated_wav_vector = orig_wav_vector[0:]
        audio_vectors[orig_wav_file] = truncated_wav_vector
        orig_wav_filenames.append(orig_wav_file)
    except:
        print('An exception occured for {}'.format(orig_wav_file))

df_testaudios['wav_file'] = orig_wav_filenames #Storing the names of all the test wav files


with open('./pre-processed/audio_vectors_test.pkl', 'wb') as f:   #Storing all the audio vectors
    pickle.dump(audio_vectors, f)



"""Extracting features from audio vectors - 8 features as shown by column headers"""

# audio_vectors_path = '{}audio_vectors_test.pkl'.format(DATA_DIR)
# audio_vectors = pickle.load(open(audio_vectors_path, 'rb')) 

columns = ['wav_file', 'sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std','ser_hap','ser_ang','ser_sad','ser_neu'] #columns represent all the features extracted added to dataframe

df_features = pd.DataFrame(columns=columns)

# print(df_testaudios.head())
# print(df_testaudios['wav_file'].head())
r = speechrec.Recognizer()

print(df_testaudios)

for index, row in tqdm(df_testaudios['wav_file'].to_frame().iterrows()):

  try:
    wav_file_name = row['wav_file']
    y = audio_vectors[wav_file_name]

    with speechrec.AudioFile(os.path.join(WAV_FILE_PATH,wav_file_name+".wav")) as source:
      audio_text = r.listen(source)
      text = r.recognize_google(audio_text)
      # print(text)
      # with open(os.path.join(TRANSCRIPTION_PATH,wav_file_name+".txt"), "w") as file1:
      #   # Writing data to a file
      #   file1.write(text)
      
      new_row = {'wav_file':wav_file_name, 'text':text, 'text_emotion_hap':None,'text_emotion_ang':None,'text_emotion_sad':None,'text_emotion_neu':None}
      text_df = text_df.append(new_row, ignore_index=True)
        # file1.close()

    feature_list = [wav_file_name]  # wav_file
    sig_mean = np.mean(abs(y))
    feature_list.append(sig_mean)  # sig_mean
    feature_list.append(np.std(y))  # sig_std

    rmse = librosa.feature.rms(y + 0.0001)[0]
    feature_list.append(np.mean(rmse))  # rmse_mean
    feature_list.append(np.std(rmse))  # rmse_std

    silence = 0
    for e in rmse:
      if e <= 0.4 * np.mean(rmse):
        silence += 1
    silence /= float(len(rmse))
    feature_list.append(silence)  # silence

    y_harmonic = librosa.effects.hpss(y)[0]
    feature_list.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)


    cl = 0.45 * sig_mean
    center_clipped = []
    for s in y:
      if s >= cl:
        center_clipped.append(s - cl)
      elif s <= -cl:
        center_clipped.append(s + cl)
      elif np.abs(s) < cl:
        center_clipped.append(0)
    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
    feature_list.append(1000 * np.max(auto_corrs)/len(auto_corrs))  # auto_corr_max (scaled by 1000)
    feature_list.append(np.std(auto_corrs))  # auto_corr_std
    feature_list.extend([0,0,0,0])
    df_features = df_features.append(pd.DataFrame(feature_list, index=columns).transpose(), ignore_index=True)
  except:
    print('Some exception occured')

df = df_features
scalar = MinMaxScaler()
df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])

df.to_csv('{}/audio_features.csv'.format(DATA_DIR), index=False) #storing all the features in a csv
text_df.to_csv('{}/audio_transcript.csv'.format(TRANSCRIPTION_PATH), index=False) 

x_test = pd.read_csv('{}/audio_features.csv'.format(DATA_DIR))
# del x_test['wav_file'] #dropping the name of the file to keep only the features for prediction

x_testset = x_test.drop(['wav_file','ser_hap','ser_sad','ser_ang','ser_neu'], axis = 1)

emotion_dict = {0:"Angry", 1:"Happy", 2:"Sad", 3:"Neutral"}

rf_classifier = pickle.load(open('{}rf_classifier.model'.format(MODEL_STORAGE_ML), 'rb'))
pred = rf_classifier.predict_proba(x_testset)
# print(pred)
for idx, val in enumerate(pred):
    # print("\nPrediction for the file {} is: \t{}".format(df['wav_file'][idx],emotion_dict[val]))
    x_test.iat[idx,9]=val[1]
    x_test.iat[idx,10]=val[0]
    x_test.iat[idx,11]=val[2]
    x_test.iat[idx,12]=val[3]

tokenizer = Tokenizer()
with open('assets/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def clean_text(data):
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    # tekenization using nltk
    data = word_tokenize(data)
    
    return data

nlp_model=pickle.load(open('./models/cnn.model', 'rb'))

class_names = ['hap', 'xxx', 'anger', 'sadness', 'neutral']
# X_test = text_df.drop('file_name','emotion')


for index,row in tqdm(text_df.iterrows()):
  flag = False
  text_file_name, text = row[0], row[1]
  words_re = re.compile("|".join(buzzwords))
  if words_re.search(text):
    flag = True
  # text=['I cannot believe I got an admit!']
  seq = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(seq, maxlen=max_seq_len)
  pred = nlp_model.predict(padded)
  # print(seq)
  # print(padded)
  # print(pred)
  # print(text_file_name+"\t\t"+class_names[np.argmax(pred)])
  text_df.iat[index,2] = pred[0][0]
  text_df.iat[index,3] = pred[0][2]
  text_df.iat[index,4] = pred[0][3]
  text_df.iat[index,5] = pred[0][4]
  text_df.iat[index,6] = flag

# print('\n')
# print(text_df)

dfinal = x_test.merge(text_df, on="wav_file")
dfinal

dfinal['combined_score_hap'] = dfinal['ser_hap']*SER_WEIGHTAGE + dfinal['text_emotion_hap']*NLP_WEIGHTAGE
dfinal['combined_score_sad'] = dfinal['ser_sad']*SER_WEIGHTAGE + dfinal['text_emotion_sad']*NLP_WEIGHTAGE
dfinal['combined_score_ang'] = dfinal['ser_ang']*SER_WEIGHTAGE + dfinal['text_emotion_ang']*NLP_WEIGHTAGE
dfinal['combined_score_neu'] = dfinal['ser_neu']*SER_WEIGHTAGE+ dfinal['text_emotion_neu']*NLP_WEIGHTAGE
# dfinal

dfinal['combined_score_neu'] = pd.to_numeric(dfinal['combined_score_neu'],errors = 'coerce')
dfinal['combined_score_sad'] = pd.to_numeric(dfinal['combined_score_sad'],errors = 'coerce')
dfinal['combined_score_hap'] = pd.to_numeric(dfinal['combined_score_hap'],errors = 'coerce')
dfinal['combined_score_ang'] = pd.to_numeric(dfinal['combined_score_ang'],errors = 'coerce')

dfinal['predicted_emotion'] = dfinal.iloc[:,-4:].idxmax(axis=1)
dfinal['predicted_emotion'] = dfinal['predicted_emotion'].str.slice(-3)

dfinal['predicted_proba'] = dfinal.iloc[:,-4:].max(axis=1)

dfinal

df.to_csv('emotion.csv')