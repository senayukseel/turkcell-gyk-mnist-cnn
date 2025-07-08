Bu projede, MNIST el yazısı rakam veriseti kullanılarak basit bir CNN modeli geliştirilmiştir. model.py dosyasında yer alan bu model, giriş olarak 28x28 boyutunda gri tonlamalı görüntüler alır ve 0'dan 9'a kadar olan rakamları sınıflandırmak üzere eğitilir. 
Eğitim işlemi tamamlandıktan sonra model .keras formatında (model.keras) kaydedilmiştir.

Eğitilen bu model, main.py dosyasında yeniden yüklenerek test edilir. Test aşamasında, oluşturduğum ve data/ klasöründe yer alan 10 adet test görseli kullanıldı.
Bu görseller, tıpkı eğitim verileri gibi gri tonlamaya çevrilir, 28x28 boyutuna yeniden boyutlandırılır ve normalleştirilir. Dosya isimlerinden gerçek etiketler otomatik olarak çıkarıldı.
Model, bu veriler üzerinde tahmin yaparak hem terminalde doğruluk oranını verip hem de tahmin sonuçlarını tahmin_sonuclari.csv dosyasına kaydetti. Ek olarak, tahmin edilen görseller matplotlib kullanılarak görselleştirilerek ve ekranda gösterilirdi.
