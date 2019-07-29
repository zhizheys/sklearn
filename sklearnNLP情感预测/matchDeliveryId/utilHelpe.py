
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class MyStringUtil():

    def removeSpecialCharacter(self,text):
        remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

        if text != None and text.strip() !='':
            return re.sub(remove_chars, ' ', text.lower())
        else:
            return ''


    def removeStopWord(self,text):

        if text == None or text.strip() =='':
            return ''

        stopWordsArray = stopwords.words('english')

        newStopWords = ['morningstar','morningstartnab','template','fund','txt', 'csv','xls','xlsx','pdf','zip','january','february','march','april',
                        'may','june','july','august','september','october','november','december']

        stopWordsArray.extend(newStopWords)

        stop_words = stopWordsArray

        word_tokens = word_tokenize(text.lower())
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        if filtered_sentence != None:
            return " ".join(filtered_sentence)
        else:
            return ''
