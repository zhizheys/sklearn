

#通过keras，进行文本预测，
import  datetime
import pandas as pd
filepath_dict = {
    'yelp':'./data.csv'}

df_list = []

for source,filepath in filepath_dict.items():
    df = pd.read_csv(filepath,names=['label','sentence'],sep=',')
    df['source'] = source
    df_list.append(df)

df =  pd.concat(df_list)

from sklearn.model_selection import train_test_split
df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

print("-----------------------------get data success")

sentence_train,sentence_test,y_train_label,y_test_label = train_test_split(sentences,y,test_size=0.25,random_state=100)

#print('train sentence is:',sentence_train)
#print('train label is:', y_train)

from sklearn.feature_extraction.text import CountVectorizer
import pickle

vectorizer = CountVectorizer()
vectorizer.fit(sentence_train)

# 保存经过fit的vectorizer 词向量模型,预测时用来对文字词向量化
feature_path = './vectorizer_matchDeliveryId_wordModel.pkl'
with open(feature_path, 'wb') as fw:
    pickle.dump(vectorizer.vocabulary_, fw)

X_train = vectorizer.transform(sentence_train)
X_test = vectorizer.transform(sentence_test)

#对标签编码
all_lable = []
all_lable.extend(y_train_label)
all_lable.extend(y_test_label)

from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
le  =  CountVectorizer()
le.fit(all_lable)

label_path = './vectorizer_matchDeliveryId_labelModel.pkl'
with open(label_path, 'wb') as fw:
    pickle.dump(le.vocabulary_, fw)

y_train =le.transform(y_train_label)
y_test = le.transform(y_test_label)

#print('X_train is:' ,X_train)
#print('X_test is:' ,X_test)

print("x train content shape: ",X_train.shape)
print("y train content shape: ",y_train.shape)
#print(X_train.shape[0])
#print(X_train.shape[1])

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10,input_dim=input_dim,activation='relu'))
#408 是标签向量化后的维度
model.add(layers.Dense(408,activation='sigmoid'))
#model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics =['accuracy'])
model.summary()

print('------------begin fit',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


history = model.fit(X_train, y_train,
				epochs=100,
				verbose=False,
				validation_data=(X_test, y_test),
				batch_size=200)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Training loss: {:.4f}".format(loss))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing loss: {:.4f}".format(loss))

print('------------end fit',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

model.save('./model_keras_matchDeliveryId.h5')

print('------------save model end')

#显示图像表格进行分析,不知道是什么原因， 目前只能在debug下才会显示图像表格
#from analystChart import plot_history

#plot_history(history)
