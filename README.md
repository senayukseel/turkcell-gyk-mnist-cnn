# CNN Digit Recognition Project

Bu proje, bilindik bir dataset olan MNIST datasetindeki el yazısı rakamları tanımak için Convolutional Neural Network (CNN) kullanarak geliştirilmiştir.
- Model, MNIST benzeri el yazısı rakamlar için eğitilmiştir
- Görseller gri tonlama olarak işlenir
- Normalizasyon için 255'e bölme işlemi uygulanır
- Çıktılar `outputs/` klasörüne `{görsel_adı}-output.png` formatında kaydedilir


### Gereksinimler

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### Kullanılan Kütüphaneler

- **TensorFlow**: CNN modeli için
- **OpenCV**: Görsel işleme
- **NumPy**: Sayısal işlemler
- **Matplotlib**: Görselleştirme

## 📊 Model Mimarisi

CNN modeli aşağıdaki katmanlardan oluşur:

- **Giriş**: 28x28x1 (gri tonlama)
- **Convolutional Katmanlar**: Feature extraction
- **MaxPooling**: Boyut azaltma
- **Dense Katmanlar**: Sınıflandırma
- **Çıkış**: 10 sınıf (0-9 rakamları)


