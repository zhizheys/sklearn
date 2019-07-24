import gensim

## 训练自己的词向量，并保存。
def trainWord2Vec(filePath):
    sentences =  gensim.models.word2vec.LineSentence(filePath) # 读取分词后的 文本
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4) # 训练模型

    model.save('./CarComment_vord2vec_100')


def testMyWord2Vec():
    # 读取自己的词向量，并简单测试一下 效果。
    inp = './CarComment_vord2vec_100'  # 读取词向量
    model = gensim.models.Word2Vec.load(inp)

    print('空间的词向量（100维）:',model['空间'])
    print('打印与空间最相近的5个词语：',model.most_similar('空间', topn=5))

    print('空间的词向量（100维）:', model['空间'])
    print('打印与空间最相近的5个词语：', model.most_similar('空间', topn=5))


if __name__ == '__main__':
    filePath = './matchDeliveryId/data.csv'
    trainWord2Vec(filePath)
    testMyWord2Vec()
    pass
