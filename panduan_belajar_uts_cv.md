# 📚 Panduan Belajar UTS Computer Vision (LKM 1–4)

> [!IMPORTANT]
> Dokumen ini adalah **rangkuman lengkap** semua sintaks dan konsep dari LKM 1 sampai 4.
> Cocok untuk dipelajari sebelum UTS yang sifatnya **menulis kode di kertas**.

---

## 📋 Daftar Isi

1. [LKM 1 — Dasar OpenCV & Pemrosesan Citra](#lkm-1)
2. [LKM 2 — Preprocessing Dataset & Face Detection](#lkm-2)
3. [LKM 3 — CNN Baseline & Transfer Learning](#lkm-3)
4. [LKM 4 — Proyek Face Recognition (Absensi Wajah)](#lkm-4)
5. [Kamus Fungsi Penting (Cheat Sheet)](#cheat-sheet)
6. [Tips Menulis Kode di Kertas Ujian](#tips-ujian)

---

## LKM 1 — Dasar OpenCV & Pemrosesan Citra {#lkm-1}

**Topik:** Pra Pemrosesan dan Pembersihan Data

### 1.1 Import Library

```python
import cv2
import numpy as np
```

> [!NOTE]
> `cv2` = OpenCV (library utama Computer Vision), `numpy` = library array/matrix numerik.

---

### 1.2 Membaca Citra dari File

```python
img = cv2.imread('cctv.jpg')     # Membaca gambar dari file
print(type(img))                 # Output: <class 'numpy.ndarray'>
print(img.shape)                 # Output: (tinggi, lebar, channel)
                                 # Contoh: (534, 800, 3) → 3 = BGR
```

> [!TIP]
> `img.shape` mengembalikan **(tinggi, lebar, jumlah_channel)**.
> Channel = 3 berarti gambar berwarna (BGR). Channel = 1 berarti grayscale.

---

### 1.3 Menampilkan Citra

```python
cv2.imshow('CCTV Image', img)    # Menampilkan gambar di window
cv2.waitKey(0)                   # Menunggu tombol keyboard ditekan
cv2.destroyAllWindows()          # Menutup semua window OpenCV
```

> [!WARNING]
> Perhatikan ejaan: `destroyAllWindows()` (dengan huruf besar A dan W).

---

### 1.4 Akses Nilai Piksel

```python
# Akses satu piksel (mengembalikan array [B, G, R])
pixel = img[100, 150]
print("Nilai piksel (BGR):", pixel)    # Contoh: [73 80 37]

# Akses satu channel saja (0=Blue, 1=Green, 2=Red)
blue = img[100, 150, 0]
print("Channel Biru:", blue)           # Contoh: 73
```

**Konsep Penting:**
- OpenCV menggunakan format **BGR** (bukan RGB!)
- Setiap channel bernilai **0–255**
- `img[y, x]` → akses piksel di baris y, kolom x
- `img[y, x, channel]` → akses channel tertentu (0=B, 1=G, 2=R)

---

### 1.5 Konversi Warna (Grayscale)

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

> [!TIP]
> Konversi populer lainnya:
> - `cv2.COLOR_BGR2RGB` → BGR ke RGB (untuk matplotlib)
> - `cv2.COLOR_BGR2HSV` → BGR ke HSV
> - `cv2.COLOR_GRAY2BGR` → Grayscale ke BGR

---

### 1.6 Resize Citra

```python
img_resized = cv2.resize(img, (224, 224))    # Resize ke lebar=224, tinggi=224
```

> [!WARNING]
> Perhatikan urutan parameter: `(lebar, tinggi)`, **bukan** `(tinggi, lebar)`.
> Ini kebalikan dari `img.shape` yang hasilnya `(tinggi, lebar, channel)`.

---

### 1.7 Rotasi Citra

```python
(h, w) = img.shape[:2]                                # Ambil tinggi & lebar
center = (w // 2, h // 2)                              # Titik tengah gambar
M = cv2.getRotationMatrix2D(center, 45, 1.0)           # Rotasi 45 derajat
rotated = cv2.warpAffine(img, M, (w, h))               # Terapkan rotasi
```

**Penjelasan Parameter `getRotationMatrix2D`:**
- `center` = titik pusat rotasi
- `45` = sudut rotasi (derajat, berlawanan jarum jam)
- `1.0` = skala (1.0 = ukuran asli)

---

### 1.8 Flipping Citra

```python
img_flip_h = cv2.flip(img, 1)     # Flip horizontal (cermin)
img_flip_v = cv2.flip(img, 0)     # Flip vertikal
img_flip_b = cv2.flip(img, -1)    # Flip keduanya (horizontal + vertikal)
```

---

### 1.9 Operasi Aritmatika Citra

```python
# Penambahan citra (mencerahkan)
bright = cv2.add(img, np.ones(img.shape, dtype=np.uint8) * 50)

# Pengurangan citra (menggelapkan)
dark = cv2.subtract(img, np.ones(img.shape, dtype=np.uint8) * 50)

# Perkalian citra (mengatur kontras)
contrast = cv2.multiply(img, np.array([1.5]))  # Kontras 1.5x
```

---

### 1.10 Penyesuaian Brightness & Contrast

```python
adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
# alpha = kontras (1.0 = normal, >1 = lebih kontras)
# beta  = brightness (0 = normal, >0 = lebih terang)
```

---

### 1.11 Gaussian Blur (Pengaburan)

```python
blurred = cv2.GaussianBlur(img, (5, 5), 0)
# (5, 5) = ukuran kernel (HARUS ganjil: 3, 5, 7, 9, ...)
# 0 = standar deviasi (otomatis dihitung dari ukuran kernel)
```

---

### 1.12 Pengasahan (Sharpening) dengan Laplacian

```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpened = cv2.convertScaleAbs(laplacian)
```

Atau dengan **kernel sharpening manual:**
```python
kernel = np.array([[0, -1, 0],
                   [-1,  5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, kernel)
```

---

### 1.13 Deteksi Tepi (Edge Detection) — Canny

```python
edges = cv2.Canny(gray, 100, 200)
# 100 = threshold bawah
# 200 = threshold atas
```

> [!TIP]
> Semakin kecil threshold → semakin banyak tepi terdeteksi.
> Semakin besar threshold → hanya tepi yang kuat yang terdeteksi.

---

### 1.14 Scaling / Penskalaan Citra

```python
scaled = cv2.resize(img, None, fx=0.5, fy=0.5)   # Perkecil 50%
scaled = cv2.resize(img, None, fx=2.0, fy=2.0)   # Perbesar 200%
```

---

### 1.15 Translasi (Pergeseran) Citra

```python
M = np.float32([[1, 0, 50],    # Geser 50 piksel ke kanan
                [0, 1, 30]])   # Geser 30 piksel ke bawah
shifted = cv2.warpAffine(img, M, (w, h))
```

---

### 1.16 Histogram Grayscale

```python
import matplotlib.pyplot as plt

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist)
plt.title('Histogram Grayscale')
plt.xlabel('Intensitas Piksel')
plt.ylabel('Jumlah Piksel')
plt.show()
```

**Penjelasan Parameter `calcHist`:**
- `[gray]` = list gambar input
- `[0]` = channel yang dihitung (0 untuk grayscale)
- `None` = mask (None = seluruh gambar)
- `[256]` = jumlah bin
- `[0, 256]` = range nilai piksel

---

### 1.17 Equalize Histogram (Pemerataan Histogram)

```python
equalized = cv2.equalizeHist(gray)
```

> [!NOTE]
> Histogram Equalization membuat distribusi intensitas piksel lebih merata,
> sehingga gambar terlihat lebih kontras. **Sering digunakan sebelum face recognition.**

---

## LKM 2 — Preprocessing Dataset & Face Detection {#lkm-2}

**Topik:** Preprocessing Dataset Foto

### 2.1 Mount Google Drive (di Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.2 Copy Dataset ke Folder Praktikum

```python
import shutil
import os

src = '/content/drive/MyDrive/dataset_wajah'
dst = '/content/praktikum/dataset'
shutil.copytree(src, dst)
```

---

### 2.3 Analisis Ukuran Gambar dalam Dataset

```python
import cv2
import os

dataset_path = '/content/praktikum/dataset'

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                print(f"{img_name}: {img.shape}")
```

---

### 2.4 Standarisasi Ukuran Gambar (224 × 224)

```python
import cv2
import os

input_dir = '/content/praktikum/dataset'
output_dir = '/content/praktikum/dataset_resized'
target_size = (224, 224)

for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    out_folder = os.path.join(output_dir, folder)
    os.makedirs(out_folder, exist_ok=True)

    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, target_size)
                cv2.imwrite(os.path.join(out_folder, img_name), resized)
```

---

### 2.5 ⭐ Face Detection dengan Haar Cascade

```python
import cv2

# 1. Muat Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 2. Baca gambar dan konversi ke grayscale
img = cv2.imread('foto.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Deteksi wajah
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,     # Faktor pengecilan gambar tiap iterasi
    minNeighbors=5       # Jumlah tetangga minimum (filter noise)
)

# 4. Gambar kotak di sekitar wajah
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

> [!IMPORTANT]
> **Ini SANGAT SERING keluar di ujian!** Hafalkan:
> - `CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')`
> - `detectMultiScale(gray, scaleFactor, minNeighbors)`
> - `cv2.rectangle(img, (x,y), (x+w, y+h), warna, ketebalan)`
> - Face detection harus menggunakan gambar **GRAYSCALE**!

---

### 2.6 ⭐ Face Cropping (Memotong Wajah)

```python
for (x, y, w, h) in faces:
    # Crop wajah dari gambar asli
    face_crop = img[y:y+h, x:x+w]

    # Resize wajah ke ukuran standar
    face_resized = cv2.resize(face_crop, (200, 200))

    # Simpan hasil crop
    cv2.imwrite(f'face_{x}.jpg', face_resized)
```

> [!TIP]
> Cropping menggunakan **slicing numpy**: `img[y:y+h, x:x+w]`
> - `y` = baris awal (atas)
> - `y+h` = baris akhir (bawah)
> - `x` = kolom awal (kiri)
> - `x+w` = kolom akhir (kanan)

---

## LKM 3 — CNN Baseline & Transfer Learning {#lkm-3}

**Topik:** Transfer Learning, Pre-Trained Model, dan Implementasinya

### 3.1 Import Library

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
```

---

### 3.2 Preprocessing Dataset dengan ImageDataGenerator

```python
datagen = ImageDataGenerator(
    rescale=1./255,           # Normalisasi piksel dari 0-255 → 0-1
    validation_split=0.2      # 20% untuk validasi
)

train_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),    # Resize otomatis
    batch_size=32,
    class_mode='categorical',  # Untuk multi-class
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

> [!NOTE]
> `rescale=1./255` → Normalisasi. Nilai piksel 0-255 menjadi 0.0-1.0.
> Ini penting agar training lebih stabil.

---

### 3.3 ⭐ Membuat Model CNN Baseline (Sederhana)

```python
model = Sequential([
    # Layer Konvolusi 1: 32 filter, ukuran kernel 3x3
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),       # Downsampling

    # Layer Konvolusi 2: 64 filter
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Layer Konvolusi 3: 128 filter
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten: ubah 2D → 1D
    Flatten(),

    # Fully Connected Layer
    Dense(128, activation='relu'),
    Dropout(0.5),                          # Dropout 50% untuk mencegah overfitting

    # Output Layer
    Dense(jumlah_kelas, activation='softmax')  # softmax untuk multi-class
])
```

> [!IMPORTANT]
> **Hafalkan arsitektur dasar CNN:**
> `Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense → Output`
>
> **Activation functions:**
> - `relu` = untuk hidden layer (paling umum)
> - `softmax` = untuk output multi-class
> - `sigmoid` = untuk output binary (2 kelas)

---

### 3.4 ⭐ Compile Model

```python
model.compile(
    optimizer='adam',                        # Optimizer populer
    loss='categorical_crossentropy',         # Loss untuk multi-class
    metrics=['accuracy']                     # Metrik evaluasi
)
```

| Situasi | Loss Function |
|---------|--------------|
| Multi-class (>2 kelas) | `categorical_crossentropy` |
| Binary (2 kelas) | `binary_crossentropy` |
| Regresi | `mse` (mean squared error) |

---

### 3.5 Training Model

```python
history = model.fit(
    train_data,
    epochs=10,                # Jumlah iterasi training
    validation_data=val_data
)
```

---

### 3.6 Plot Hasil Training

```python
# Plot Akurasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

### 3.7 ⭐ Transfer Learning dengan MobileNetV2

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# 1. Load pre-trained model TANPA top layer
base_model = MobileNetV2(
    weights='imagenet',         # Bobot dari ImageNet
    include_top=False,          # Buang fully connected layer teratas
    input_shape=(224, 224, 3)
)

# 2. Freeze base model (JANGAN update bobotnya)
base_model.trainable = False

# 3. Tambahkan layer custom di atas
x = base_model.output
x = GlobalAveragePooling2D()(x)          # Pooling global
x = Dense(128, activation='relu')(x)     # Hidden layer
output = Dense(jumlah_kelas, activation='softmax')(x)  # Output

# 4. Buat model baru
model_tl = Model(inputs=base_model.input, outputs=output)

# 5. Compile
model_tl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Training
history_tl = model_tl.fit(train_data, epochs=10, validation_data=val_data)
```

> [!IMPORTANT]
> **Konsep Transfer Learning yang harus dipahami:**
> 1. `include_top=False` → Tidak ikut-sertakan layer klasifikasi asli (ImageNet 1000 kelas)
> 2. `base_model.trainable = False` → **Freeze** bobot model, hanya train layer tambahan
> 3. Kita menambahkan layer sendiri di atas untuk tugas klasifikasi kita
> 4. Transfer Learning **lebih cepat dan lebih akurat** daripada CNN dari nol

---

### 3.8 Evaluasi & Perbandingan Model

```python
# Evaluasi model
loss, accuracy = model.evaluate(val_data)
print(f"Baseline — Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

loss_tl, accuracy_tl = model_tl.evaluate(val_data)
print(f"Transfer Learning — Loss: {loss_tl:.4f}, Accuracy: {accuracy_tl:.4f}")
```

---

## LKM 4 — Proyek Face Recognition (Absensi Wajah) {#lkm-4}

**Topik:** Project Computer Vision — Face Detection & Recognition

### 4.1 Arsitektur Proyek (7 File)

| File | Fungsi |
|------|--------|
| `users.csv` | Menyimpan data user (id, nama, nim, folder) |
| `attendance.csv` | Menyimpan log kehadiran |
| `main.py` | File utama, GUI dengan Tkinter |
| `register.py` | Logika registrasi wajah baru |
| `login.py` | Logika login dengan pengenalan wajah |
| `attendance.py` | Logika pencatatan kehadiran |
| `camera_utils.py` | Utilitas membuka kamera |
| `train.py` | Training model LBPH Face Recognizer |

---

### 4.2 ⭐ Membuka Kamera (camera_utils.py)

```python
import cv2
import numpy as np

def open_camera():
    cap = cv2.VideoCapture(0)   # 0 = kamera default laptop
    if cap.isOpened():
        return cap
    return None
```

**Loop membaca frame dari kamera:**
```python
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()    # ret = True/False, frame = gambar
    if not ret:
        break

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):   # Tekan 'q' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
```

> [!NOTE]
> `cap.read()` mengembalikan **dua nilai**:
> - `ret` (boolean) → apakah frame berhasil dibaca
> - `frame` (numpy array) → gambar frame saat ini

---

### 4.3 ⭐ LBPH Face Recognizer — Training (train.py)

```python
import cv2
import os
import numpy as np

def train_model():
    # 1. Buat recognizer LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []     # List gambar wajah
    labels = []    # List label (user_id)

    dataset_dir = "dataset"

    # 2. Loop setiap folder user
    for user_folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, user_folder)
        if not os.path.isdir(folder_path):
            continue

        # Ambil user_id dari nama folder (misal "user_001" → 1)
        user_id = int(user_folder.split("_")[1])

        # 3. Loop setiap gambar dalam folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.equalizeHist(img)    # Equalize histogram

            faces.append(img)
            labels.append(user_id)

    # 4. Training
    recognizer.train(faces, np.array(labels))

    # 5. Simpan model
    recognizer.save("models/trainer.yml")
    return True
```

> [!IMPORTANT]
> **LBPH (Local Binary Pattern Histogram):**
> - `cv2.face.LBPHFaceRecognizer_create()` → membuat recognizer
> - `recognizer.train(faces, labels)` → training dengan data wajah + label
> - `recognizer.save(path)` → simpan model
> - `recognizer.read(path)` → load model yang sudah disimpan
> - `recognizer.predict(face)` → prediksi wajah → return `(user_id, confidence)`

---

### 4.4 ⭐ Face Recognition — Login (login.py)

```python
import cv2

# 1. Load model dan detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/trainer.yml")

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 2. Baca frame dari kamera
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, 1.3, 5)

# 3. Untuk setiap wajah yang terdeteksi
for (x, y, w, h) in faces:
    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))   # Crop & resize
    face = cv2.equalizeHist(face)                          # Equalize

    # 4. Prediksi
    user_id, confidence = recognizer.predict(face)

    # 5. Cek confidence (semakin kecil = semakin mirip)
    if confidence < 65:
        label = "Dikenali"              # Wajah cocok
    else:
        label = "Tidak Dikenali"        # Wajah tidak cocok

    # 6. Gambar kotak dan label
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
```

> [!TIP]
> **Confidence pada LBPH:**
> - Semakin **kecil** nilainya = semakin **mirip** dengan data training
> - Threshold umum: `< 50-70` dianggap cocok
> - `confidence < 65` berarti wajah dikenali

---

### 4.5 Registrasi Wajah (register.py) — Alurnya

```python
# 1. Deteksi wajah dari frame kamera
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, 1.3, 5)

# 2. Crop, resize, dan simpan wajah
for (x, y, w, h) in faces:
    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    cv2.imwrite(f"dataset/user_001/{count}.jpg", face)
    count += 1

# 3. Setelah cukup foto, simpan data ke CSV
import csv
with open("users.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([user_id, nama, nim, folder_name])

# 4. Training ulang model
train_model()
```

---

### 4.6 Pencatatan Kehadiran (attendance.py)

```python
import csv
from datetime import datetime

def mark_attendance(user_id, nama, nim):
    today = datetime.now().strftime("%Y-%m-%d")     # Format tanggal
    now_time = datetime.now().strftime("%H:%M:%S")  # Format waktu

    # Cek apakah sudah absen hari ini
    with open("attendance.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["id"] == str(user_id) and row["tanggal"] == today:
                return False    # Sudah absen

    # Tulis data absensi
    with open("attendance.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, nama, nim, today, now_time, "Hadir"])

    return True
```

---

### 4.7 Menampilkan Teks di Gambar

```python
cv2.putText(
    frame,                          # Gambar
    "Hello World",                  # Teks
    (10, 30),                       # Posisi (x, y)
    cv2.FONT_HERSHEY_SIMPLEX,      # Font
    0.7,                            # Ukuran font
    (0, 255, 0),                    # Warna (BGR)
    2                               # Ketebalan
)
```

---

## Kamus Fungsi Penting (Cheat Sheet) {#cheat-sheet}

### OpenCV (cv2)

| Fungsi | Kegunaan | Contoh |
|--------|----------|--------|
| `cv2.imread(path)` | Membaca gambar | `img = cv2.imread('a.jpg')` |
| `cv2.imshow(title, img)` | Menampilkan gambar | `cv2.imshow('Img', img)` |
| `cv2.imwrite(path, img)` | Menyimpan gambar | `cv2.imwrite('out.jpg', img)` |
| `cv2.cvtColor(img, code)` | Konversi warna | `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| `cv2.resize(img, (w,h))` | Mengubah ukuran | `r = cv2.resize(img, (224,224))` |
| `cv2.flip(img, code)` | Membalik gambar | `f = cv2.flip(img, 1)` |
| `cv2.GaussianBlur(img, k, s)` | Mengaburkan | `b = cv2.GaussianBlur(img,(5,5),0)` |
| `cv2.Canny(gray, t1, t2)` | Deteksi tepi | `e = cv2.Canny(gray, 100, 200)` |
| `cv2.rectangle(img, p1, p2, c, t)` | Menggambar kotak | `cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)` |
| `cv2.putText(...)` | Menulis teks | Lihat contoh di atas |
| `cv2.equalizeHist(gray)` | Equalize histogram | `eq = cv2.equalizeHist(gray)` |
| `cv2.calcHist(...)` | Menghitung histogram | Lihat contoh di atas |
| `cv2.getRotationMatrix2D(...)` | Matriks rotasi | `M = cv2.getRotationMatrix2D(c,45,1.0)` |
| `cv2.warpAffine(img, M, (w,h))` | Terapkan transformasi | `r = cv2.warpAffine(img, M, (w,h))` |
| `cv2.VideoCapture(0)` | Buka kamera | `cap = cv2.VideoCapture(0)` |
| `cap.read()` | Baca frame | `ret, frame = cap.read()` |
| `cap.release()` | Tutup kamera | `cap.release()` |

### Haar Cascade (Face Detection)

| Fungsi | Kegunaan |
|--------|----------|
| `cv2.CascadeClassifier(path)` | Membuat detector |
| `detector.detectMultiScale(gray, sf, mn)` | Mendeteksi wajah |

### LBPH Face Recognizer

| Fungsi | Kegunaan |
|--------|----------|
| `cv2.face.LBPHFaceRecognizer_create()` | Membuat recognizer |
| `recognizer.train(faces, labels)` | Training model |
| `recognizer.save(path)` | Simpan model |
| `recognizer.read(path)` | Muat model |
| `recognizer.predict(face)` | Prediksi → `(id, confidence)` |

### Keras / TensorFlow (CNN)

| Fungsi | Kegunaan |
|--------|----------|
| `Sequential([...])` | Membuat model sequential |
| `Conv2D(filters, kernel, activation)` | Layer konvolusi |
| `MaxPooling2D(pool_size)` | Layer pooling |
| `Flatten()` | Meratakan 2D → 1D |
| `Dense(units, activation)` | Fully connected layer |
| `Dropout(rate)` | Dropout (cegah overfitting) |
| `model.compile(optimizer, loss, metrics)` | Compile model |
| `model.fit(data, epochs, validation_data)` | Training model |
| `model.evaluate(data)` | Evaluasi model |
| `MobileNetV2(weights, include_top, input_shape)` | Load pre-trained model |
| `GlobalAveragePooling2D()` | Global average pooling |

---

## Tips Menulis Kode di Kertas Ujian {#tips-ujian}

### ✅ Yang WAJIB Diperhatikan

1. **Indentasi** — Python sangat ketat. Beri jarak menjorok yang **jelas** untuk isi `for`, `if`, `def`, `class`
2. **Tanda titik dua (`:`)** — Jangan lupa setelah `if`, `for`, `while`, `def`, `class`
3. **Tanda kurung** — `print()` harus pakai kurung (Python 3)
4. **Import** — Selalu tulis import di awal: `import cv2`, `import numpy as np`

### 🎯 Strategi Mengerjakan

1. **Tulis import dulu** di baris paling atas
2. **Tulis kerangka/alur** dulu (komentar), baru isi kode
3. Jika lupa nama fungsi persis, **tulis yang mendekati** + beri komentar penjelasan
4. **Jangan panik** kalau lupa parameter opsional — tulis parameter wajib saja

### 📝 Prediksi Soal yang Kemungkinan Keluar

| # | Topik Soal | Fungsi Kunci |
|---|-----------|-------------|
| 1 | Baca & tampilkan gambar | `imread`, `imshow`, `waitKey` |
| 2 | Konversi ke grayscale | `cvtColor`, `COLOR_BGR2GRAY` |
| 3 | Resize gambar | `resize` |
| 4 | Deteksi wajah Haar Cascade | `CascadeClassifier`, `detectMultiScale` |
| 5 | Crop wajah | `img[y:y+h, x:x+w]` |
| 6 | Gambar kotak di wajah | `rectangle` |
| 7 | Blur/Sharpen/Edge detection | `GaussianBlur`, `Canny` |
| 8 | Arsitektur CNN sederhana | `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` |
| 9 | Transfer Learning | `MobileNetV2`, `include_top=False`, `trainable=False` |
| 10 | LBPH Recognizer | `create`, `train`, `predict` |
| 11 | Buka kamera & baca frame | `VideoCapture`, `cap.read()` |
| 12 | Histogram | `calcHist`, `equalizeHist` |

### 🔥 Contoh Soal Latihan (Kerjakan di Kertas!)

**Soal 1:** Tulis kode untuk membaca gambar `wajah.jpg`, konversi ke grayscale, deteksi wajah, gambar kotak hijau di wajah, dan tampilkan hasilnya.

**Jawaban:**
```python
import cv2

img = cv2.imread('wajah.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Hasil', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

**Soal 2:** Tulis kode untuk membuat model CNN sederhana dengan 2 layer konvolusi untuk mengklasifikasikan 5 orang.

**Jawaban:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

---

**Soal 3:** Tulis kode untuk load model LBPH, baca frame kamera, deteksi wajah, dan kenali wajah tersebut.

**Jawaban:**
```python
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('models/trainer.yml')

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    face = cv2.equalizeHist(face)
    user_id, confidence = recognizer.predict(face)

    if confidence < 65:
        print(f"User {user_id} dikenali")
    else:
        print("Tidak dikenali")

cap.release()
```

---

> [!CAUTION]
> **Jangan hanya membaca dokumen ini!** Ambil kertas kosong dan coba tulis ulang kode-kode di atas **tanpa melihat**. Ini cara paling efektif untuk mempersiapkan UTS tulis kode di kertas.
