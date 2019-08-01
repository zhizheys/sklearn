
from matchDeliveryIdKerasTextCNN.prepareData import PrepareData
from matchDeliveryIdKerasTextCNN.createMatchDeliveryModelByKeras import  MatchDeliveryIdModelByKeras
from matchDeliveryIdKerasTextCNN.predictByKerasMappingDeliveryId import MyPredictByKerasMappingDeliveryId

if __name__ == '__main__':

    print('-------------start app')
    prepareData = PrepareData()
    prepareData.createStandardFile()
    #
    model = MatchDeliveryIdModelByKeras()
    model.createModel()

    # --准确率： 0.963(sigmoid)  --0.999(init.xlsx)  --0.965(init_new.xlsx)
    # id PDG0015788
    # sender = 'M.Ilmansyah@bahana.co.id'
    # subject = 'NAV Bahana Trailblazer Fund and Bahana Provident Fund'
    # fileName = 'NAV BTF & BPF.xls'

    # --准确率： 0.029  --0.96 --error
    # id PDN0008563
    sender = 'niclas.roken@alcur.se'
    subject = 'ALCUR FONDER performance updates.xls'
    fileName = 'ALCUR FONDER performance updates.xls'

    # --准确率： 0.026234388  --0.993 --0.030623268
    # id PDG0005141
    # sender = 'lexb@hcmirae.com'
    # subject = '每日净值_2019年06月27日_华宸未来基金管理有限公司'
    # fileName = '每日净值_2019年06月27日_华宸未来基金管理有限公司.xlsx'

    # predictItem = MyPredictByKerasMappingDeliveryId()
    # predictLabel,accuracy = predictItem.startPredict(sender,subject,fileName)
    #
    # print('predict label is: ', predictLabel)
    # print('predict accuracy is: ', accuracy)
