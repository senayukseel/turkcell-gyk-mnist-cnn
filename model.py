# 07.07.25 Pazartesi
import tensorflow as tf
import matplotlib.pyplot as plt

# MNIST dataet -> 0-9 arası rakamların olduğu bir veriseti
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalizasyon ->RGB kanallarının 0-255 aralığından 0-1 aralığına çekilmesidir.
X_train = X_train / 255
X_test = X_test / 255

# CNN'in input formatı -> (örnek sayısı, genişlik, yükseklik, kanal sayısı)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape((-1, 28, 28, 1)) 

plt.figure(figsize=(10, 10))

for i in range(10):
    plt.subplot(10, 10, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap = "gray")
    plt.axis("off")
plt.suptitle("İlk 10 görüntü")
plt.show()

# CNN
# Sequential -> katmanları sırayla eklediğimiz bir yapı
model = tf.keras.models.Sequential(
    [

        # Kernel size -> 3x3 (görüntü üzerinde 3x3 filtrelerle dolaş)
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)), # görüntü 2x2 boyutundaki alanların max değerlerini alarak görüntüyü küçültür -> ilk tarama sonrası bilgiyi özetle
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"), # daha karmaşık bilgileir öğrenebilmek için
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),  # çok boyutlu çıktıyu tek boyutlu (1D) çıktıya çevirir.
        tf.keras.layers.Dense(128, activation="relu"), # 128 nöronlu bir katman. nöron -> karar mekanizmasıdır.
        tf.keras.layers.Dense(10, activation="softmax") # verimizde 10 sınıf olduğu için 10 nöronlu bir katman kullanıldı
        # softmax -> 10 rakam arasında tahmin gücen skoru en yüksek olanı seçer. 10 tahmin yapar ve en en yüksek değeri seçer.
    ]
)

model.summary()

# modelin genel parametrelerini belirleyip eğitmeye hazır hale getirir. 
# optimizer -> modelin eğitimi sırasında kullanılacak optimizasyon algoritması
# loss -> modelin eğitimi sırasında kullanılacak loss fonksiyonu
# metrics = modelin eğitimi sırasında kullanılacak metrikler
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# epoch -> veriyi baştan sonra kaçkere göreceğiz?
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

model.save("model.keras")
