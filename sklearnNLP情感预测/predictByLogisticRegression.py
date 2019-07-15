

#根据前面预测的模型，对数据进行情感分析

import pickle
from sklearn.feature_extraction.text import CountVectorizer

#load 词向量模型
fileVector = open('./models/vectorizer_wordModel.pkl','rb')
vectorizer = pickle.load(fileVector)
fileVector.close()

#load 情感分类预测模型
filePredict = open('./models/logicPredictModel.pkl','rb')
predictModel = pickle.load(filePredict)
filePredict.close()


testStr=['very good, i like it']
loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vectorizer)
x_testStr = loaded_vec.transform(testStr)

prediction=predictModel.predict(x_testStr)

print("predict result is :",prediction[0])

if 1== prediction[0]:
    print("正面评价")
else:
    print("负面评价")
