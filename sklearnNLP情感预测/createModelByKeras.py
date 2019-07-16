

#通过keras，进行文本预测，

import pandas as pd
filepath_dict = {
    'yelp':'./yelp_labelled.txt',
    'amazon':'./amazon_cells_labelled.txt',
    'imdb':'./imdb_labelled.txt'}

df_list = []

for source,filepath in filepath_dict.items():
    df = pd.read_csv(filepath,names=['sentence','label'],sep='\t')
    df['source'] = source
    df_list.append(df)

df =  pd.concat(df_list)

from sklearn.model_selection import train_test_split
df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentence_train,sentence_test,y_train,y_test = train_test_split(sentences,y,test_size=0.25,random_state=100)

#print('train sentence is:',sentence_train)
#print('train label is:', y_train)

from sklearn.feature_extraction.text import CountVectorizer
import pickle

vectorizer = CountVectorizer()
vectorizer.fit(sentence_train)

# 保存经过fit的vectorizer 词向量模型,预测时用来对文字词向量化
feature_path = './models/vectorizer_wordModel.pkl'
with open(feature_path, 'wb') as fw:
    pickle.dump(vectorizer.vocabulary_, fw)

X_train = vectorizer.transform(sentence_train)
X_test = vectorizer.transform(sentence_test)

#print('X_train is:' ,X_train)
#print('X_test is:' ,X_test)

#print(X_train.shape)
#print(X_train.shape[0])
print(X_train.shape[1])

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10,input_dim=input_dim,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics =['accuracy'])
model.summary()


history = model.fit(X_train, y_train,
				epochs=100,
				verbose=False,
				validation_data=(X_test, y_test),
				batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Training loss: {:.4f}".format(loss))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing loss: {:.4f}".format(loss))



#show chart
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
