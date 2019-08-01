

#通过keras，进行文本预测，
import  datetime
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers,Model
from keras.layers import Dense,Conv1D,MaxPooling1D,Embedding,Input,Flatten,concatenate,Dropout

class MatchDeliveryIdModelByKeras():


    def createModel(self):

        print("----------------begin create model")

        filepath = './data.csv'
        df = pd.read_csv(filepath,names=['label','sentence'],sep=',',encoding='utf-8')
        sentences = df['sentence'].fillna(' ')
        y = df['label'].fillna(' ')


        from sklearn.model_selection import train_test_split

        print("-----------------------------get data success")
        print('-------------------------begin word to encode')

        encoder = LabelEncoder()
        encoded_int_Y = encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse=False)
        encoded_int_Y = encoded_int_Y.reshape(len(encoded_int_Y), 1)
        onehot_encoded_Y = onehot_encoder.fit_transform(encoded_int_Y)



        tokenizer = Tokenizer()  # 创建一个Tokenizer对象
        # fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
        tokenizer.fit_on_texts(sentences)
        self.vocab = tokenizer.word_index  # 得到每个词的编号
        x_train, x_test, y_train, y_test = train_test_split(sentences,onehot_encoded_Y, test_size=0.25)
        # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
        x_train_word_ids = tokenizer.texts_to_sequences(x_train)
        x_test_word_ids = tokenizer.texts_to_sequences(x_test)
        # 序列模式
        # 每条样本长度不唯一，将每条样本的长度设置一个固定值
        x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=500)  # 将超过固定值的部分截掉，不足的在最前面用0填充
        x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=500)


        #--------------------------

        self.textCNN_model(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test)
        #self.textCNN_model(x_train_padded_seqs, y_train_padded_seqs, x_test_padded_seqs, y_test_padded_seqs)

    def textCNN_model(self,x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):
        print("-------------begin create text cnn model")

        main_input = Input(shape=(500,), dtype='float64')
        # 词嵌入（使用预训练的词向量）
        embedder = Embedding(len(self.vocab) + 1, 300, input_length=500, trainable=False)
        embed = embedder(main_input)
        # 词窗大小分别为3,4,5
        cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=48)(cnn1)
        cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=47)(cnn2)
        cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=46)(cnn3)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(0.2)(flat)

        #main_output = Dense(3, activation='softmax')(drop)

        input_dim = x_train_padded_seqs.shape[1]
        label_dim = y_train.shape[1]
        main_output = Dense(output_dim=label_dim,input_dim=input_dim,activation='softmax')(drop)

        model = Model(inputs=main_input, outputs=main_output)


        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
        #model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=10)


        model.fit(x_train_padded_seqs, y_train, batch_size=150, epochs=10)

        # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
        result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
        result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
        y_predict = list(map(str, result_labels))

        print('准确率', metrics.accuracy_score(y_test, y_predict))
        print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

