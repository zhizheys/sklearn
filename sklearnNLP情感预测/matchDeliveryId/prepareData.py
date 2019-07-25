

#read data from excel

import pandas as pd
import xlrd
from matchDeliveryId.utilHelpe import MyStringUtil

class PrepareData():

    def createStandardFile(self):
        print("-----------begin prepare data")
        #按列读取数据
        # wbFile = xlrd.open_workbook("./initial.xlsx")
        # mysheet  = wbFile.sheet_by_name('Sheet1')
        # senderColumn = mysheet.col_values(0)
        # subjectColumn = mysheet.col_values(1)
        # fileNameColumn = mysheet.col_values(2)
        # deliveryIdColumn = mysheet.col_values(3)

        #print(senderColumn)

        #按照单元格读取数据
        df=pd.read_excel("./initial_new.xlsx",sheet_name='Sheet1')
        targetData=[]
        myStringUtil = MyStringUtil()

        for j in df.index.values:
            row_data = df.ix[j,['sender','subject','fileName','deliveryId']].to_dict()
            sender = str(row_data['sender']).lower() if row_data['sender'] !=None else ''
            subject = str(row_data['subject']).lower() if row_data['subject'] !=None else ''
            fileName = str(row_data['fileName']).lower() if row_data['fileName'] !=None else ''
            deliveryId = str(row_data['deliveryId']).lower() if row_data['deliveryId'] !=None else ''



            sender = myStringUtil.removeSpecialCharacter(sender)
            sender = myStringUtil.removeStopWord(sender)

            subject = myStringUtil.removeSpecialCharacter(subject)
            subject = myStringUtil.removeStopWord(subject)

            fileName = myStringUtil.removeSpecialCharacter(fileName)
            fileName = myStringUtil.removeStopWord(fileName)

            fileInfo = sender + ' ' + subject + ' ' + fileName

            if fileInfo != None and fileInfo.strip() != '' and len(fileInfo.strip()) >0 and deliveryId != None and deliveryId.strip() != '' and len(deliveryId.strip()) > 0:
                temp={'fileInfo':fileInfo.strip(),'deliveryId':deliveryId.strip()}
                targetData.append(temp)


        #写入csv

        path = "./data.csv"
        # f = open(path, "w")
        # csv_head = 'deliveryId fileInfo'
        # #create header
        # f.write(csv_head)
        #

        #create content
        dataContent =[]
        for k in targetData:
            data_row = [k['deliveryId'], k['fileInfo']]
            dataContent.append(data_row)

        import csv


        with open(path,"w",newline='',encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            #先写入columns_name
            #writer.writerow(["deliveryId","fileInfo"])
            #写入多行用writerows
            writer.writerows(dataContent)


        print("----------------prepare data end")




