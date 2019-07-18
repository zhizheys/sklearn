
import utilHelpe

a='_T-Rowe IMD_eff20190321_rel2019-03-22_133349.xls'
b = utilHelpe.removeSpecialCharacter(a)
print(b)
c = utilHelpe.removeStopWord(b)
print(c)


# import nltk
# nltk.download('punkt')