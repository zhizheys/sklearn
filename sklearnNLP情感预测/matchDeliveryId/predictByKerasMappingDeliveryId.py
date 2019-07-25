
#根据前面预测的模型，对数据进行情感分析

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import  load_model
from matchDeliveryId.utilHelpe import MyStringUtil


class MyPredictByKerasMappingDeliveryId():

    def predictInfo(self,fileInfo):

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

    def createContentInfo(self,strArray):
        contentInfo=''
        myStringUtil = MyStringUtil()
        if strArray != None and len(strArray) > 0:
            for j in strArray:
                j = myStringUtil.removeSpecialCharacter(j)
                j = myStringUtil.removeStopWord(j)
                contentInfo = contentInfo + ' ' + j

        return contentInfo.strip()

    def createContentInfo2(self,sender,subject,fileName):
        contentText=''
        myStringUtil = MyStringUtil()

        sender = myStringUtil.removeSpecialCharacter(sender)
        sender = myStringUtil.removeStopWord(sender)

        subject = myStringUtil.removeSpecialCharacter(subject)
        subject = myStringUtil.removeStopWord(subject)

        fileName = myStringUtil.removeSpecialCharacter(fileName)
        fileName = myStringUtil.removeStopWord(fileName)

        fileInfo = sender + ' ' + subject + ' ' + fileName

        if fileInfo != None and fileInfo.strip() != '':
            contentText=fileInfo.lower().strip()

        return  contentText

    def startPredict(self,sender,subject,fileName):

        contentArray = [sender,subject,fileName]
        fileInfo = self.createContentInfo(contentArray)
        predictLabel,accuracy =self.predictInfo(fileInfo)

        return predictLabel,accuracy

