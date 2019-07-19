


#根据前面预测的模型，对数据进行情感分析

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import  load_model
import utilHelpe

def predictInfo(fileInfo):

    targetLabel = 'No found'
    maxSimilar=0

    #load 词向量模型
    fileVector = open('./vectorizer_matchDeliveryId_wordModel.pkl','rb')
    vectorizer = pickle.load(fileVector)
    fileVector.close()

    #load mapping delivery id model
    model = load_model('./model_keras_matchDeliveryId.h5')

    testStr=[fileInfo]
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vectorizer)
    x_testStr = loaded_vec.transform(testStr)

    prediction=model.predict(x_testStr)
    prediction_class=model.predict_classes(x_testStr)

    maxSimilar = prediction[0].max()
    # print('max similar is ', maxSimilar)
    # print('max similar label Id is: ',prediction[0].argmax())
    # print('class is',prediction_class)


    #load 词向量模型
    labelFilePath = open('./vectorizer_matchDeliveryId_labelModel.pkl','rb')
    labelDic = pickle.load(labelFilePath)
    labelFilePath.close()


    loaded_label_vec = CountVectorizer(decode_error="replace", vocabulary=labelDic)
    loaded_label_vec.fit(labelDic)

    #get dic info
    keys = loaded_label_vec.vocabulary.keys()
    for k in keys:
        if loaded_label_vec.vocabulary[k] == prediction_class[0]:
            targetLabel = k
            break

    return  targetLabel,maxSimilar


def createContentInfo(strArray):
    contentInfo=''
    if strArray != None and len(strArray) > 0:
        for j in strArray:
            j = utilHelpe.removeSpecialCharacter(j)
            j = utilHelpe.removeStopWord(j)
            contentInfo = contentInfo + ' ' + j

    return contentInfo.strip().lower()


if __name__ == '__main__':

    sender ='ViewsEmailNotification.DoNotReply@jpmorgan.com'
    subject = 'Schedule T-Rowe IMD  Completed on Thu 16-May-2019 13:15:30 AEST'
    fileName = '_T-Rowe IMD_eff20190515_rel2019-05-16_130815.xls'
    contentArray = [sender,subject,fileName]

    fileInfo = createContentInfo(contentArray)
    print("ccc -------",fileInfo)

    predictLabel,accuracy =predictInfo(fileInfo)
    print('predict label is: ',predictLabel)
    print('predict accuracy is: ', accuracy)