#Veri Setini İçeri Aktar
import pandas as pd

data = pd.read_csv("spam.csv" , encoding="latin-1")

data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],  axis=1)

data.columns = ["label", "text"]

#Veri Keşfi
 ## print(data.isna().sum())

#Text Preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")

text = list(data["text"])

lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub("[^A-Za-z]"," ",text[i])
    r = r.lower()
    r = nltk.word_tokenize(r)
    r = [word for word in r if word not in stopwords.words("english")]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = " ".join(r)
    corpus.append(r)

data["text2"] = corpus

# Train_Test_Split (%67 - %33)
from sklearn.model_selection import train_test_split

y = data["label"]
X = data["text2"]

X_train , X_test, Y_train ,Y_test = train_test_split(X ,y,
                                    random_state = 42,
                                    shuffle = True,
                                    test_size = 0.33)

# Feature Extraction

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train )

# Classifier Training
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train_cv , Y_train)

x_test_cv = cv.transform(X_test)

#prediction
prediction = dt.predict(x_test_cv)

from sklearn.metrics import confusion_matrix

c_matrix = confusion_matrix(Y_test, prediction)

Percent = [(c_matrix[0,0] + c_matrix[1,1]) /sum(sum(c_matrix))]
print(f"Accuracy : {Percent}")
