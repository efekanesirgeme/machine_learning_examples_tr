# Sklearnde hazır olan 'datasets' kütüphanesini yüklüyoruz.
from sklearn import datasets
# Sklearndeki SupportVectorMachines(svm) sınıflandırma(classification) yöntemlerinden SVC ve LinearSVC'yi yüklüyoruz.
from sklearn.svm import SVC, LinearSVC
# Sklearnden SGDClassifier modelini seçiyoruz.
from sklearn.linear_model import SGDClassifier
# Sklearnden NaiveBayes yöntemini yüklüyoruz.
from sklearn.naive_bayes import GaussianNB
# Datamızı Train&Test olarak ikiye bölebilmek için train&test_split modelini yüklüyoruz.
from sklearn.model_selection import train_test_split
# Datamızı DataFrame'e dönüştürmek için pandas kütüphanesini yüklüyoruz.
import pandas as pd
# Eğittiğimiz modelimizi kaydetmek ve kaydedilen model üzerinden test işlemleri yapmak adına pickle kütüphanesini yüklüyoruz.
import pickle
# Test modelimizin accuracy_score'unu hesaplayabilmek için sklearn'den accuracy_score metodunu yüklüyoruz.
from sklearn.metrics import accuracy_score
# Çalıştırdığımız kodda zamansal hesaplamalar yapabilmek için time modülünü yüklüyoruz.
import time

print("Data Yükleniyor")
time.sleep(0.5)
iris = datasets.load_iris()
print("Data Yüklendi")
time.sleep(1.2)

print("DataFrame dönüştürülüyor")
time.sleep(0.5)
df = pd.DataFrame(iris.data)
X = df
Y = iris.target
print("DataFrame dönüştürüldü")
time.sleep(1.2)

print("Data Bölünüyor")
time.sleep(0.5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
print("Data Bölündü")
time.sleep(1.2)

print("Modeller seçiliyor")
time.sleep(0.5)
model = SVC(gamma='auto')
model2 = LinearSVC(loss='squared_hinge', dual=True, tol=0.0001, C=3.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=10, random_state=42, max_iter=5000)
model3 = SGDClassifier()
model4 = GaussianNB()
print("Modeller seçildi")
time.sleep(1.2)

print("Fit ediliyor")
time.sleep(0.5)
model.fit(X_train, Y_train)
model2.fit(X_train, Y_train)
model3.fit(X_train, Y_train)
model4.fit(X_train, Y_train)
print("Fit edildi")
time.sleep(1.2)

print("Predict Ediliyor")
time.sleep(0.5)
pred = model.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)
pred4 = model4.predict(X_test)
print("Predict edildi")
time.sleep(1.2)

print("Sonuçlar hesaplanıyor lütfen bekleyiniz.")
time.sleep(0.5)
toplam_saniye = 4
print("Sonuçlar hesaplandı ekrana bastırılıyor")
time.sleep(1.2)

while toplam_saniye != 1:
    toplam_saniye -= 1
    print("Kalan:", toplam_saniye, "saniye")
    time.sleep(1)

print("*****************SONUÇLAR*****************\n")
print("SVC:", model.score(X_train, Y_train))
print("LinearSVC:", model2.score(X_train, Y_train))
print("SGDClassifier", model3.score(X_train, Y_train))
print("GaussianNB", model4.score(X_train, Y_train))
print("\n")
print("******************************************")

