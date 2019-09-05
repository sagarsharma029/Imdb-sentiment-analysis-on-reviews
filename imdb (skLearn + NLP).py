from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

df = pd.read_csv('imdb.txt',
            sep='\t',header=None)
columns = ['review', 'sentiment']
df.columns = columns

x = df['review'].values
y = df['sentiment'].values

for i in range(len(x)):
    x[i] = word_tokenize(x[i].lower())

stop_words = stopwords.words('english')

stop_words.extend([',','.','-'])

stop_words = set(stop_words)

for i in range(len(x)):
    x[i] = list(set(x[i]) - stop_words)

lemmatizer = WordNetLemmatizer()

for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = lemmatizer.lemmatize(x[i][j], pos='v')

for i in range(len(x)):
    x[i] = ' '.join(x[i])

vect = TfidfVectorizer()

matrix = vect.fit_transform(x)

x = matrix.toarray()

nb = GaussianNB()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)

accuracy_score(y_test,y_pred)

logistic = LogisticRegression()

logistic.fit(x_train,y_train)

y_pred = logistic.predict(x_test)

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
