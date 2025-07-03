import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Muat model MobileNetV2 pra-latih
base_model = MobileNetV2(weights='imagenet', include_top=False)  # Tidak sertakan lapisan atas

# Tambahkan lapisan klasifikasi baru di atas model pra-latih
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # 2 kelas: Male dan Female

# Buat model akhir
model = Model(inputs=base_model.input, outputs=predictions)

# Membekukan lapisan base_model agar tidak terlatih lagi
for layer in base_model.layers:
    layer.trainable = False

# Kompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ubah warna dari BGR ke RGB
    img = cv2.resize(img, (224, 224))  # Ubah ukuran gambar ke 224x224
    img_array = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Preprocessing untuk MobileNetV2
    return img_array

# Muat file deteksi wajah dari OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi deteksi wajah
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Fungsi untuk prediksi gender dari gambar
def predict_gender(image):
    faces = detect_face(image)
    if len(faces) > 0:
        # Ambil area wajah pertama yang terdeteksi
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]
        img_array = preprocess_image(face_image)
        predictions = model.predict(img_array)
        gender = "Male" if np.argmax(predictions) == 0 else "Female"
        return gender
    else:
        return "No face detected"

# Akses kamera lokal
cap = cv2.VideoCapture(0)  # Buka kamera lokal (0 adalah default)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Prediksi gender pada setiap frame
    gender = predict_gender(frame)

    # Tampilkan hasil prediksi pada frame
    cv2.putText(frame, f"Gender: {gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tampilkan frame dengan hasil prediksi
    cv2.imshow("Gender Prediction", frame)

    # Keluar dari loop jika tekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
