import numpy as np
import pandas as pd
import unicodedata
import re
import string
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import LancasterStemmer


df = pd.read_csv('../labeled.csv')
df_train = df.copy()

# lower casing
df_train['text']= df_train['text'].apply(lambda x: x.lower())

# remove punctuation
df_train['text']= df_train['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# remove accented characters
def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text
df_train["text"] = df_train["text"].apply(lambda x: remove_accented_chars(x))

# remove special characters
def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)
df_train["text"] = df_train["text"].apply(lambda x: remove_accented_chars(x))

# remove numbers
for i, row in df_train.iterrows():
  tmp = re.sub(r'\d', ' ', row['text'])
  tmp = " ".join(tmp.split())
  df_train.at[i, 'text'] = tmp


# remove stop words
nltk.download('stopwords')
from nltk.corpus import stopwords

english_stop_words_list=stopwords.words('english')
def rem_stpwrds(text):
    return " ".join([word for word in str(text).split() if word not in english_stop_words_list])
df_train["text"] = df_train["text"].apply(lambda x: rem_stpwrds(x))

# TODO - spell correction
 
# stemming
lc = LancasterStemmer()

def stem(text):
    return " ".join([lc.stem(word) for word in text.split()])

df_train['text'] = df_train['text'].apply(stem)

# SVM
x_data = df_train['text']
y_data = df_train['sport']

x_train , x_test , y_train , y_test = train_test_split(x_data , y_data , stratify=y_data, test_size=0.2, random_state=42 )

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced')))
])

pipeline.fit(x_train, y_train)

def predict(text):
    test_text = stem(text)
    predicted_label = pipeline.predict([test_text])
    for i, j in sorted(zip(pipeline.predict_proba([test_text])[0],pipeline.classes_)):
        print(j.ljust(15, ' '), i)
    return predicted_label[0]
