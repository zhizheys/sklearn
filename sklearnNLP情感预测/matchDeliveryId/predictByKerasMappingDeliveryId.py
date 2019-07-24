


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

    return contentInfo.strip()

def createContentInfo2(sender,subject,fileName):
    contentText=''
    sender = utilHelpe.removeSpecialCharacter(sender)
    sender = utilHelpe.removeStopWord(sender)

    subject = utilHelpe.removeSpecialCharacter(subject)
    subject = utilHelpe.removeStopWord(subject)

    fileName = utilHelpe.removeSpecialCharacter(fileName)
    fileName = utilHelpe.removeStopWord(fileName)

    fileInfo = sender + ' ' + subject + ' ' + fileName

    if fileInfo != None and fileInfo.strip() != '':
        contentText=fileInfo.lower().strip()

    return  contentText

if __name__ == '__main__':

    #--准确率： 0.963(sigmoid)  --0.999(init.xlsx)  --0.965(init_new.xlsx)
    #id PDG0015788
    # sender = 'M.Ilmansyah@bahana.co.id'
    # subject = 'NAV Bahana Trailblazer Fund and Bahana Provident Fund'
    # fileName = 'NAV BTF & BPF.xls'

    #--准确率： 0.029  --0.96 --error
    #id PDN0008563
    # sender = 'swlee@educatorsfinancialgroup.ca'
    # subject = '[Not Virus Scanned] April 30, 2019 Educators Mutual Funds'
    # fileName = 'Morningstar Educators 04-30-2019 fund report.xlsx'

    # --准确率： 0.026234388  --0.993 --0.030623268
    # id PDG0005141
    sender = 'Ntebogeng.Mogagabe@investecmail.com'
    subject = 'Discovery and Investec Fund Distributions - Mar 2019'
    fileName = 'Investec Distributions - Mar 2019.xlsx'

    contentArray = [sender,subject,fileName]

    fileInfo = createContentInfo(contentArray)
    predictLabel,accuracy =predictInfo(fileInfo)
    print('predict label is: ',predictLabel)
    print('predict accuracy is: ', accuracy)

