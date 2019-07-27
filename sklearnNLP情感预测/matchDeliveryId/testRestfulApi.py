
import pandas as pd
import xlrd
import requests
import json

class TestRestfulApi:
    def readExcelData(self):
        # 按照单元格读取数据
        df = pd.read_excel("./initial_new.xlsx", sheet_name='Sheet1')
        targetData = []

        for j in df.index.values:
            row_data = df.ix[j, ['sender', 'subject', 'fileName', 'deliveryId']].to_dict()
            sender = str(row_data['sender']).lower() if row_data['sender'] != None else ''
            subject = str(row_data['subject']).lower() if row_data['subject'] != None else ''
            fileName = str(row_data['fileName']).lower() if row_data['fileName'] != None else ''
            deliveryId = str(row_data['deliveryId']).lower() if row_data['deliveryId'] != None else ''
            fileInfo = sender + ' ' + subject + ' ' + fileName

            if fileInfo != None and fileInfo.strip() != '' and len(
                    fileInfo.strip()) > 0 and deliveryId != None and deliveryId.strip() != '' and len(
                    deliveryId.strip()) > 0:
                temp = {'sender': sender,'subject': subject,'fileName': fileName, 'deliveryId': deliveryId.strip()}
                targetData.append(temp)

        return targetData

    def post(self,sender,subject,fileName,initDeliveryId):
        result=''
        url=''
        json_data = dict(
            sender=sender,
            subject=subject,
            fileName=fileName
        )

        headerContent ={
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }

        response = requests.post(url, headers=headerContent, data=json.dumps(json_data))
        print(response.status_code)

        if(response.status_code==200):
            resultContent = response.text
            resultObject = json.loads(resultContent)
            predictDeliveryId = resultObject['deliveryId']

            if predictDeliveryId == None or predictDeliveryId == '':
                result='deliveryIdIsNull'
            else:
                if initDeliveryId == predictDeliveryId:
                    result='equal'
                else:
                    result='unequal'
        else:
            result='postError'

        return result

    def get(self):
        headerContent = {
            'Accept': 'application/json'
        }
        url='https://www.tianqiapi.com/api/?version=v1&cityid=101110101'
        response = requests.get(url,headers=headerContent)
        print(response.status_code)
        print(response.text)
        return "回调发送成功"

    def aa(self):
        print('bbbb')


if __name__ == '__main__':
    print('-----------start api')
    test = TestRestfulApi()
    dataArray = test.readExcelData()

    for item in dataArray:
        sender =item['sender']
        subject = item['subject']
        fileName = item['fileName']
        deliveryId = item['deliveryId']

        result =  test.post(sender,subject,fileName,deliveryId)
        item['result'] = result

    for item in dataArray:
        result =item['result']
        print('result is: ',result)