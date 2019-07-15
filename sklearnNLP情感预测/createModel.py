import pandas as pd
filepath_dict = {
    'yelp':'./yelp_labelled.txt',
    'amazon':'./amazon_cells_labelled.txt',
    'imdb':'./imdb_labelled.txt'}

df_list = []

for source,filepath in filepath_dict.items():
    df = pd.read_csv(filepath,names=['sentence','label'],sep='\t')
    df['source'] = source
    df_list.append(df)

df =  pd.concat(df_list)

from sklearn.model_selection import train_test_split
df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentence_train,sentence_test,y_train,y_test = train_test_split(sentences,y,test_size=0.25,random_state=100)

print('train sentence is:',sentence_train)
print('train label is:', y_train)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sentence_train)

X_train = vectorizer.transform(sentence_train)
X_text = vectorizer.transform(sentence_test)

#print('X_train is:' ,X_train)
#print('X_test is:' ,X_text)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train,y_train)
score = classifier.score(X_text,y_test)

print("train Logist score is: ",score)

testStr=['very good, i like it']
x_testStr = vectorizer.transform(testStr)

prediction=classifier.predict(x_testStr)

print("predict result is :",prediction)



