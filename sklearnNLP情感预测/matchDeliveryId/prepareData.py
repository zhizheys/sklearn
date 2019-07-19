

#read data from excel

import pandas as pd
import xlrd
import utilHelpe

#按列读取数据
# wbFile = xlrd.open_workbook("./initial.xlsx")
# mysheet  = wbFile.sheet_by_name('Sheet1')
# senderColumn = mysheet.col_values(0)
# subjectColumn = mysheet.col_values(1)
# fileNameColumn = mysheet.col_values(2)
# deliveryIdColumn = mysheet.col_values(3)

#print(senderColumn)

#按照单元格读取数据
df=pd.read_excel("./initial.xlsx",sheet_name='Sheet1')
targetData=[]

for j in df.index.values:
    row_data = df.ix[j,['sender','subject','fileName','deliveryId']].to_dict()
    sender = str(row_data['sender']) if row_data['sender'] !=None else ''
    subject = str(row_data['subject']) if row_data['subject'] !=None else ''
    fileName = str(row_data['fileName']) if row_data['fileName'] !=None else ''
    deliveryId = row_data['deliveryId']

    sender = utilHelpe.removeSpecialCharacter(sender)
    sender = utilHelpe.removeStopWord(sender)

    subject = utilHelpe.removeSpecialCharacter(subject)
    subject = utilHelpe.removeStopWord(subject)

    fileName = utilHelpe.removeSpecialCharacter(fileName)
    fileName = utilHelpe.removeStopWord(fileName)

    fileInfo = sender + ' ' + subject + ' ' + fileName

    if fileInfo != None and fileInfo.strip() != '' and deliveryId != None and deliveryId.strip() != '':
        temp={'fileInfo':fileInfo.lower().strip(),'deliveryId':deliveryId.lower().strip()}
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


print("----------------end")




