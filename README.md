# 📱 Text Classification ile SMS Spam Tespiti

Bu proje, **SMS mesajlarının spam (istenmeyen mesaj)** olup olmadığını **Text Classification (Metin Sınıflandırması)** yöntemiyle tahmin etmeyi amaçlamaktadır.
Proje kapsamında **Doğal Dil İşleme (NLP)** teknikleri ve **Makine Öğrenmesi algoritmaları** kullanılarak bir sınıflandırma modeli oluşturulmuştur.

---

## 🚀 Proje Adımları

### 1. Veri Seti

Kullanılan veri seti: **spam.csv**

Veri setinde iki temel sütun bulunmaktadır:

* `label`: Mesajın türü (“ham” = normal, “spam” = istenmeyen mesaj)
* `text`: Mesajın içeriği

İlk olarak gereksiz sütunlar kaldırılmış ve kolon isimleri sadeleştirilmiştir

```python
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data.columns = ["label", "text"]
```

---

### 2. Metin Ön İşleme (Text Preprocessing)

Metinler, modelin anlayabileceği forma getirilmiştir.
Bu aşamada yapılan işlemler:

* Özel karakterlerin temizlenmesi
* Küçük harfe dönüştürme
* Tokenization (kelimeye ayırma)
* Stopword’lerin kaldırılması
* Lemmatization (kelimeleri kök haline getirme)

```python
r = re.sub("[^A-Za-z]", " ", text[i])
r = r.lower()
r = nltk.word_tokenize(r)
r = [word for word in r if word not in stopwords.words("english")]
r = [lemmatizer.lemmatize(word) for word in r]
```

Sonuçlar `text2` adlı yeni bir sütuna kaydedilmiştir.

---

### 3. Eğitim ve Test Verisine Ayırma

Veri seti, %67 eğitim ve %33 test olacak şekilde ikiye ayrılmıştır:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

---

### 4. Özellik Çıkarımı (Feature Extraction)

Metin verileri, **Bag of Words (BoW)** yöntemiyle sayısal forma dönüştürülmüştür:

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
```

---

### 5. Model Eğitimi

Sınıflandırıcı olarak **Decision Tree Classifier** kullanılmıştır:

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_cv, Y_train)
```

---

### 6. Tahmin ve Başarı Oranı

Model test verisi üzerinde denenmiş ve doğruluk oranı hesaplanmıştır:

```python
prediction = dt.predict(x_test_cv)
c_matrix = confusion_matrix(Y_test, prediction)
Percent = [(c_matrix[0,0] + c_matrix[1,1]) / sum(sum(c_matrix))]
print(f"Accuracy : {Percent}")
```

---

## 📊 Sonuçlar

Model, test verisi üzerinde **yaklaşık %X doğruluk oranı** elde etmiştir
(çıktı çalıştırıldığı ortama göre değişebilir).

---

## 🧰 Kullanılan Kütüphaneler

* **pandas** → Veri okuma ve düzenleme
* **nltk** → Metin işleme (tokenization, stopword, lemmatization)
* **scikit-learn** → Model eğitimi, test ayrımı ve metrik hesaplama

---

## 💡 Geliştirme Fikirleri

* CountVectorizer yerine **TF-IDF Vectorizer** denenebilir.
* **Naive Bayes**, **Logistic Regression** veya **Random Forest** gibi farklı modeller karşılaştırılabilir.
* Daha fazla veriyle modelin başarısı artırılabilir.
* Model Flask veya Streamlit ile web arayüzüne dönüştürülebilir.

---

## 📚 Özet

Bu proje, **Metin Sınıflandırma (Text Classification)** yaklaşımı kullanarak SMS mesajlarının spam olup olmadığını tespit eden temel bir NLP uygulamasıdır.
Proje, makine öğrenmesi ve metin işleme alanlarında temel bir örnek teşkil eder.

---

