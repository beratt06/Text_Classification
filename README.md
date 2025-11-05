# 📱 SMS Spam Tespiti (Text Classification)

Bu projede, SMS mesajlarının **spam (istenmeyen mesaj)** olup olmadığını tahmin eden basit bir **metin sınıflandırma** modeli geliştirdim.
Amaç, gelen bir mesajın içeriğine göre onu "spam" ya da "normal" olarak ayırmaktı.

---

## 🔹 1. Veri Seti

Projede **spam.csv** adlı veri setini kullandım.
Veri setinde iki temel sütun bulunuyor:

* `label`: Mesajın türü (spam veya ham)
* `text`: Mesajın içeriği

İlk olarak gereksiz sütunları sildim ve isimleri düzenledim:

```python
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data.columns = ["label", "text"]
```

---

## 🔹 2. Metin Ön İşleme

Bu kısımda mesajların içeriğini modele uygun hale getirdim.
Yani gereksiz karakterleri temizledim, küçük harfe çevirdim, stopword’leri (önemsiz kelimeleri) çıkardım ve kelimeleri kök haline getirdim.

Kısaca yapılan işlemler:

* Semboller ve sayılar kaldırıldı
* Tüm harfler küçültüldü
* İngilizce stopword’ler çıkarıldı
* Kelimeler lemmatize edildi (kök haline getirildi)

Bu işlemlerden sonra temizlenmiş metinleri `text2` adında yeni bir sütuna ekledim.

---

## 🔹 3. Veriyi Eğitim ve Test Olarak Ayırma

Veriyi %67 eğitim ve %33 test olacak şekilde ayırdım:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

Böylece modelin öğrenmesi ve sonrasında test edilmesi için iki ayrı kısım oluşturuldu.

---

## 🔹 4. Özellik Çıkarımı

Metinleri modele verebilmek için sayısal değerlere dönüştürmem gerekiyordu.
Bunun için **CountVectorizer** yöntemini kullandım. Bu yöntem, her kelimenin metinde kaç defa geçtiğini sayıyor:

```python
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
```

---

## 🔹 5. Model Eğitimi

Model olarak **Decision Tree Classifier (Karar Ağacı)** kullandım.
Bu algoritma, veriye göre dallanarak karar verir ve sonunda sınıfı (spam veya ham) tahmin eder.

```python
dt = DecisionTreeClassifier()
dt.fit(X_train_cv, Y_train)
```

---

## 🔹 6. Tahmin ve Sonuç

Eğitimden sonra test verisiyle modelin doğruluğunu ölçtüm:

```python
prediction = dt.predict(x_test_cv)
c_matrix = confusion_matrix(Y_test, prediction)
Percent = [(c_matrix[0,0] + c_matrix[1,1]) / sum(sum(c_matrix))]
print(f"Accuracy : {Percent}")
```

Modelin doğruluk oranı yaklaşık **%X civarındaydı** (çalıştığı ortama göre değişebilir).

---

## 🔹 Kullanılan Kütüphaneler

* **pandas** – Veri okuma ve düzenleme
* **nltk** – Metin işleme (stopword, lemmatization vs.)
* **scikit-learn** – Model eğitimi ve test işlemleri

---

## 💡 İleride Yapılabilecekler

* **TF-IDF Vectorizer** kullanarak kelimelerin önemini daha iyi hesaplamak
* Farklı algoritmalar (Naive Bayes, Random Forest vb.) denemek
* Web arayüzü oluşturup kullanıcıdan SMS metni alarak tahmin yapmak

---

## 🧾 Özet

Bu proje, basit bir **Doğal Dil İşleme (NLP)** uygulaması olarak SMS mesajlarını analiz edip spam olup olmadığını tahmin ediyor.
Hem metin ön işleme hem de makine öğrenmesi tarafında temel ama öğretici bir örnek oldu.

---
