
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import numpy as np
import re
import codecs
from nltk.tokenize import RegexpTokenizer
import nltk
from normalization_morningstar import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from numpy import array
from numpy import argmax


# In[4]:


import keras
print(keras.__version__)
import tensorflow as tf
print(tf.__version__)


# In[2]:


questions = pd.read_csv(r'D:\WorkItem\GPP\AIPredictDeliveryId.csv',encoding='gb18030') 
questions.columns=['Send', 'Subject', 'FileName','DeliveryIds'] 
questions["FullText"] = questions["Send"] + ' '+ questions["Subject"] + ' '+questions["FileName"] 
questions.head()


# In[3]:


def standardize_text(df, text_field):
    df[text_field] =df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r'.xls', '')
    df[text_field] = df[text_field].str.replace(r'.xlsx', '')
    df[text_field] = df[text_field].str.replace(r'.csv', '')
    df[text_field] = df[text_field].str.replace(r'.txt', '')
    df[text_field] = df[text_field].str.replace(r'.pdf', '')
    df[text_field] = normalize_corpus(df[text_field],lemmatize = False)
    df[text_field] = df[text_field].str.replace(r'\d+', '')
    return df


# In[4]:


questions = standardize_text(questions, "FullText")
questions.to_csv(r'D:\WorkItem\GPP\AIPredictDeliveryId_Clear.csv')
tokenizer_text = RegexpTokenizer(r'\w+')
clear_questons = pd.read_csv(r'D:\WorkItem\GPP\AIPredictDeliveryId_Clear.csv')
clear_questons["Tokens"] = clear_questons['FullText'].apply(tokenizer_text.tokenize)


# In[5]:


all_words = [word for tokens in clear_questons['Tokens'] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clear_questons["Tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))


# In[6]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Sentence length')
plt.ylabel('Number of sentences')
plt.hist(sentence_lengths)
plt.show()


# In[8]:


Y = clear_questons["DeliveryIds"].tolist()
encoder = LabelEncoder()
encoded_int_Y = encoder.fit_transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)
encoded_int_Y = encoded_int_Y.reshape(len(encoded_int_Y), 1)
onehot_encoded_Y= onehot_encoder.fit_transform(encoded_int_Y)


# In[9]:


X_train,X_test,y_train,y_test = train_test_split(questions["FullText"].tolist(),onehot_encoded_Y,test_size=0.1)


# In[10]:


tokenizer = Tokenizer(num_words= len(VOCAB))  
tokenizer.fit_on_texts(X_train)


# In[11]:


print (len(VOCAB))


# In[12]:


word2id = tokenizer.word_index


# In[13]:


vocab_size = len(word2id)
print (vocab_size)


# In[14]:


print('Vocabulary Sample:', list(word2id.items()))


# In[15]:


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_full = sequence.pad_sequences(X_train_seq, maxlen=max(sentence_lengths))
X_test_full = sequence.pad_sequences(X_test_seq, maxlen=max(sentence_lengths))
#Y_train_label = np.array(y_train)
#Y_train_label_Binary = to_categorical(Y_train_label)
Y_train_label = y_train
Y_test_label = y_test
 


# In[16]:


DICT_SIZE = len(VOCAB)
MAX_SENTENSE_SIZE = max(sentence_lengths)
print(DICT_SIZE)
print(MAX_SENTENSE_SIZE)


# In[17]:


def text_cnn(maxlen=30, max_features=2885, embed_size=300):
    # Inputs
    #model = Sequential()
    
    comment_seq = Input(shape=[maxlen], name='x_seq')
    #model.add(comment_seq)
    print (maxlen)
    print (max_features)
    
    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size,input_length=maxlen)(comment_seq)
    #model.add(emb_comment)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(output_dim=407,input_dim=32, activation='relu')(out)

    output = Dense(output_dim=407,input_dim=32, activation='softmax')(output)

    model = Model([comment_seq], output)
    #model.add(output)
    #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


# In[18]:


model = text_cnn(MAX_SENTENSE_SIZE,DICT_SIZE,)
model.summary()
batch_size = 128
epochs = 20
model.fit(X_train_full, Y_train_label,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)


# In[57]:


loss, accu = model.evaluate(X_train_full,Y_train_label)
print(loss, accu) 


# In[62]:


json_model = model.to_json()
open('suggust_deliveryid.jason','w').write(json_model)
model.save_weights('suggust_delivery_weights.h5')


# In[24]:


pred = model.predict(X_train_full)


# In[48]:


print(X_train_full[205])


# In[49]:


print(X_train_full[205])
print (X_train[205])


# In[50]:


init_lables = encoder.inverse_transform(argmax(pred[205:206]))
print (init_lables)


# In[51]:


str_sampe ='navreports ntrscom controlfida delta defensive ucits  controlfida delta defensive ucits'
test_str_sample = 'navreport  controlfida defensive ucits  controlfida delta defensive ucits'
data = ['navreport  controlfida defensive ucits  controlfida delta defensive ucits']
values = array(data)
#print(values)
X_train_seq_1 = tokenizer.texts_to_sequences(values)
#print(X_train_seq_1)
X_train_full_1 = sequence.pad_sequences(X_train_seq_1, maxlen=max(sentence_lengths))
#print(X_train_full_1)
pred_1 = model.predict(X_train_full_1)
#print(pred_1)
init_lables_1 = encoder.inverse_transform(argmax(pred_1[0:1]))
print (init_lables_1)


# In[69]:


def output_user_traindata(input_str, sentence_length = 30):
    input_str_list = [input_str]
    X_train_seq = tokenizer.texts_to_sequences(array(input_str_list))
    X_train_full = sequence.pad_sequences(X_train_seq,sentence_length)
    return X_train_full


# In[70]:


X_train_full_2 = output_user_traindata(test_str_sample,max(sentence_lengths))


# In[71]:


pred_2 = model.predict(X_train_full_2)
init_lables_2 = encoder.inverse_transform(argmax(pred_2[0:1]))
print (init_lables_2)


# In[53]:


test_list = [str_sampe]
print (test_list)


# In[64]:


from keras.models import model_from_json


# In[65]:


# load json and create model
json_file = open('suggust_deliveryid.jason', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("suggust_delivery_weights.h5")
print("Loaded model from disk")


# In[66]:


loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
score = loaded_model.evaluate(X_test_full,Y_test_label, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[200]:


from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data= [1,3,2,0,3,2,2,1,0,1]
data= array(data)
print(data)
# one hot encode
encoded= to_categorical(data)
print(encoded)
# invert encoding
inverted= argmax(encoded[5])
print(inverted)


# In[201]:


from numpy import argmax
# define input string
data = 'hello world'
print(data)
# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)
# one hot encode
onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(alphabet))]
	letter[value] = 1
	onehot_encoded.append(letter)
print(onehot_encoded)
# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)


# In[31]:


from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)


# In[72]:


from keras.preprocessing.text import Tokenizer

docs = ["A heart that",
         "full up like",
         "a landfill",
        "no surprises",
        "and no alarms"
         "a job that slowly"
         "Bruises that",
         "You look so",
         "tired happy",
         "no alarms",
        "and no surprises"]
docs_train = docs[:7]
docs_test = docs[7:]
# EXPERIMENT 1: FIT  TOKENIZER ONLY ON TRAIN
T_1 = Tokenizer()
T_1.fit_on_texts(docs_train)  # only train set
encoded_train_1 = T_1.texts_to_sequences(docs_train)
encoded_test_1 = T_1.texts_to_sequences(docs_test)
print("result for test 1:\n%s" %(encoded_test_1,))

# EXPERIMENT 2: FIT TOKENIZER ON BOTH TRAIN + TEST
T_2 = Tokenizer()
T_2.fit_on_texts(docs)  # both train and test set
encoded_train_2 = T_2.texts_to_sequences(docs_train)
encoded_test_2 = T_2.texts_to_sequences(docs_test)
print("result for test 2:\n%s" %(encoded_test_2,))


# In[73]:


print (docs_train)


# In[75]:


print (docs_test)


# In[76]:


T_1.fit_on_texts(docs_train)  # only train set
encoded_train_1 = T_1.texts_to_sequences(docs_train)


# In[77]:


print (encoded_train_1)


# In[78]:


encoded_test_1 = T_1.texts_to_sequences(docs_test)


# In[79]:


print (encoded_test_1)


# In[80]:


T_2 = Tokenizer()
T_2.fit_on_texts(docs)  # both train and test set
encoded_train_2 = T_2.texts_to_sequences(docs_train)


# In[81]:


print (encoded_train_2)


# In[82]:


encoded_test_2 = T_2.texts_to_sequences(docs_test)


# In[83]:


print (encoded_test_2)


# In[84]:


import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



# In[85]:


# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer_test = pickle.load(handle)


# In[86]:


print (tokenizer_test)


# In[88]:


print (max(sentence_lengths))


# In[94]:


str_sampe ='navreports ntrscom controlfida delta defensive ucits  controlfida delta defensive ucits'
test_str_sample = 'navreport  controlfida defensive ucits  controlfida delta defensive ucits'
data = ['navreport  controlfida defensive ucits  controlfida delta defensive ucits']
values = array(data)
#print(values)
X_train_seq_1_fromfile = tokenizer_test.texts_to_sequences(values)
print(X_train_seq_1_fromfile)
X_train_full_1_fromfile = sequence.pad_sequences(X_train_seq_1_fromfile, maxlen=max(sentence_lengths))
print(X_train_full_1_fromfile)
pred_1_fromfile = model.predict(X_train_full_1_fromfile)
print(pred_1_fromfile)
init_lables_1_fromfile = encoder.inverse_transform(argmax(pred_1_fromfile[0:1]))
print (argmax(pred_1_fromfile[0:1]))
print (init_lables_1_fromfile)


# In[95]:


model_fromfile = model_from_json(open('suggust_deliveryid.jason').read())
model_fromfile.load_weights('suggust_delivery_weights.h5')

probabilities = model_fromfile.predict(X_train_full_1_fromfile)
print (probabilities)


# In[96]:


Y = clear_questons["DeliveryIds"].tolist()


# In[97]:


print (Y)


# In[99]:


from sklearn.preprocessing import LabelBinarizer



# In[100]:


encoder_Binary = LabelBinarizer()
transfomed_label = encoder_Binary.fit_transform(Y)


# In[101]:


print (transfomed_label)


# In[109]:


print (len(transfomed_label[0:1][0]))


# In[110]:


original_label = encoder_Binary.inverse_transform(transfomed_label[0:1])


# In[111]:


print(original_label)


# In[112]:


with open('label_delivery.pickle', 'wb') as handle:
    pickle.dump(encoder_Binary, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[113]:


with open('label_delivery.pickle', 'rb') as handle:
    encoder_Binary_test = pickle.load(handle)


# In[114]:


print(encoder_Binary_test)

