
import re

def removeSpecialCharacter(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

    if text != None and text.strip() !='':
        return re.sub(remove_chars, ' ', text)
    else:
        return ''

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def removeStopWord(text):

    if text == None or text.strip() =='':
        return ''

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    if filtered_sentence != None:
        return " ".join(filtered_sentence)
    else:
        return ''


