# 📝 Prediksi Soal UTS Computer Vision (LKM 1–3)
# Lengkap: Syntax Rumpang • Penjelasan Kode • Prediksi Output

---

## 🔷 BAGIAN A — LKM 1: Dasar OpenCV & Pemrosesan Citra

---

### 📖 Teori Singkat

**Apa itu OpenCV?**
OpenCV (Open Computer Vision) adalah library Python untuk memproses gambar dan video. Di Python, kita import dengan `import cv2`.

**Apa itu Citra/Image?**
Dalam komputer, gambar = **array 3 dimensi** (numpy array) dengan format `(tinggi, lebar, channel)`.
- Gambar **berwarna**: 3 channel → **B**lue, **G**reen, **R**ed (BGR, bukan RGB!)
- Gambar **grayscale**: 1 channel → hitam putih (0=hitam, 255=putih)

**Mengapa BGR bukan RGB?**
OpenCV secara historis menggunakan format BGR. Ini penting karena kalau kamu pakai matplotlib untuk menampilkan, harus dikonversi ke RGB dulu.

---

### ✏️ SOAL TIPE 1: Melengkapi Syntax Rumpang

---

**Soal 1.** Lengkapi kode berikut untuk membaca gambar dan menampilkan tipe serta dimensinya:

```python
import _____(a)_____
import numpy as np

img = cv2._____(b)_____('cctv.jpg')
print(_____(c)_____(img))
print(img._____(d)_____)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) cv2
(b) imread
(c) type
(d) shape
```
**Penjelasan:**
- `cv2.imread()` = membaca gambar dari file
- `type(img)` = menampilkan tipe data → hasilnya `<class 'numpy.ndarray'>`
- `img.shape` = menampilkan dimensi → hasilnya misalnya `(534, 800, 3)`
</details>

---

**Soal 2.** Lengkapi kode untuk menampilkan gambar di window:

```python
cv2._____(a)_____('CCTV Image', img)
cv2._____(b)_____(0)
cv2._____(c)_____()
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) imshow
(b) waitKey
(c) destroyAllWindows
```
**Penjelasan:**
- `imshow(judul, gambar)` = menampilkan gambar di window dengan judul tertentu
- `waitKey(0)` = menunggu tombol keyboard ditekan (0 = tunggu selamanya)
- `destroyAllWindows()` = menutup semua window OpenCV
</details>

---

**Soal 3.** Lengkapi kode untuk mengakses nilai piksel pada posisi baris 100, kolom 150:

```python
pixel = img[_____(a)_____, _____(b)_____]
print("Nilai piksel (BGR):", pixel)

blue = img[100, 150, _____(c)_____]
print("Channel Biru:", blue)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) 100
(b) 150
(c) 0
```
**Penjelasan:**
- `img[baris, kolom]` atau `img[y, x]` = akses piksel
- Channel: `0 = Blue`, `1 = Green`, `2 = Red`
- Hasil `pixel` contoh: `[73 80 37]` → B=73, G=80, R=37
</details>

---

**Soal 4.** Lengkapi kode untuk mengkonversi gambar berwarna ke grayscale:

```python
gray = cv2._____(a)_____(img, cv2._____(b)_____)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) cvtColor
(b) COLOR_BGR2GRAY
```
**Penjelasan:**
- `cvtColor` = convert color, mengubah format warna
- `COLOR_BGR2GRAY` = dari BGR (berwarna) ke Grayscale (hitam putih)
- Setelah konversi, `gray.shape` hanya 2 dimensi: `(tinggi, lebar)` tanpa channel
</details>

---

**Soal 5.** Lengkapi kode untuk mengubah ukuran gambar menjadi 224×224 piksel:

```python
img_resized = cv2._____(a)_____(img, (_____(b)_____, _____(c)_____))
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) resize
(b) 224
(c) 224
```
**Penjelasan:**
- `cv2.resize(gambar, (lebar, tinggi))`
- ⚠️ Urutan parameter: **(lebar, tinggi)**, bukan (tinggi, lebar)!
- Ini kebalikan dari `img.shape` yang mengembalikan (tinggi, lebar, channel)
</details>

---

**Soal 6.** Lengkapi kode untuk membalik gambar secara horizontal:

```python
flipped = cv2._____(a)_____(img, _____(b)_____)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) flip
(b) 1
```
**Penjelasan:**
- `cv2.flip(img, 1)` = flip horizontal (cermin kiri-kanan)
- `cv2.flip(img, 0)` = flip vertikal (cermin atas-bawah)
- `cv2.flip(img, -1)` = flip keduanya
</details>

---

**Soal 7.** Lengkapi kode untuk melakukan Gaussian Blur:

```python
blurred = cv2._____(a)_____(img, (_____(b)_____, _____(c)_____), 0)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) GaussianBlur
(b) 5
(c) 5
```
**Penjelasan:**
- `GaussianBlur(gambar, ukuran_kernel, sigma)`
- Ukuran kernel **harus angka ganjil**: (3,3), (5,5), (7,7), dll.
- Semakin besar kernel = semakin buram
- `0` = sigma dihitung otomatis
</details>

---

**Soal 8.** Lengkapi kode untuk deteksi tepi menggunakan Canny:

```python
edges = cv2._____(a)_____(gray, _____(b)_____, _____(c)_____)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) Canny
(b) 100
(c) 200
```
**Penjelasan:**
- `cv2.Canny(gambar_grayscale, threshold_bawah, threshold_atas)`
- Threshold bawah = batas minimum kekuatan tepi
- Threshold atas = batas minimum untuk tepi yang kuat
- Input **harus grayscale**
</details>

---

**Soal 9.** Lengkapi kode untuk merotasi gambar 45 derajat:

```python
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2._____(a)_____(center, _____(b)_____, 1.0)
rotated = cv2._____(c)_____(img, M, (w, h))
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) getRotationMatrix2D
(b) 45
(c) warpAffine
```
**Penjelasan:**
- `getRotationMatrix2D(pusat, sudut, skala)` = membuat matriks transformasi rotasi
- `warpAffine(gambar, matriks, (lebar, tinggi))` = menerapkan transformasi
- Sudut positif = berlawanan arah jarum jam
</details>

---

**Soal 10.** Lengkapi kode untuk menghitung dan equalize histogram:

```python
hist = cv2._____(a)_____([gray], [0], None, [256], [0, 256])
equalized = cv2._____(b)_____(gray)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) calcHist
(b) equalizeHist
```
**Penjelasan:**
- `calcHist` = menghitung distribusi intensitas piksel
- `equalizeHist` = meratakan distribusi histogram agar kontras lebih baik
- Keduanya membutuhkan input **grayscale**
</details>

---

### ✏️ SOAL TIPE 2: Jelaskan Kode Berikut

---

**Soal 11.** Jelaskan fungsi setiap baris kode berikut:

```python
img = cv2.imread('foto.jpg')           # Baris 1
print(img.shape)                        # Baris 2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Baris 3
cv2.imwrite('gray.jpg', gray)           # Baris 4
```

<details>
<summary>🔑 Jawaban</summary>

- **Baris 1:** Membaca (load) gambar dari file `foto.jpg` ke dalam variabel `img` sebagai numpy array
- **Baris 2:** Menampilkan dimensi gambar dalam format `(tinggi, lebar, channel)`. Contoh output: `(480, 640, 3)` artinya tinggi 480px, lebar 640px, 3 channel warna (BGR)
- **Baris 3:** Mengkonversi gambar dari format BGR (berwarna) menjadi Grayscale (hitam putih). Hasilnya disimpan di variabel `gray`
- **Baris 4:** Menyimpan gambar grayscale ke file baru bernama `gray.jpg`. Fungsi `imwrite` = image write (tulis gambar ke disk)
</details>

---

**Soal 12.** Jelaskan apa yang dilakukan kode berikut dan apa output-nya:

```python
import cv2
import numpy as np

img = cv2.imread('gambar.jpg')
bright = cv2.add(img, np.ones(img.shape, dtype=np.uint8) * 50)
```

<details>
<summary>🔑 Jawaban</summary>

**Penjelasan:**
- Baris 1-2: Import library OpenCV dan NumPy
- Baris 4: Membaca gambar dari file
- Baris 5: Menambahkan nilai 50 ke setiap piksel pada setiap channel gambar

**Apa yang terjadi:**
- `np.ones(img.shape, dtype=np.uint8) * 50` = membuat array berisi angka 50 dengan ukuran sama dengan gambar
- `cv2.add()` = menjumlahkan kedua array
- Hasilnya: gambar menjadi **lebih terang** karena nilai setiap piksel bertambah 50
- Nilai akan di-clip di 255 (tidak melebihi 255) berkat `cv2.add()`
</details>

---

**Soal 13.** Jelaskan perbedaan `cv2.add()` dengan operator `+` biasa:

```python
hasil_a = cv2.add(img, np.ones(img.shape, dtype=np.uint8) * 200)
hasil_b = img + np.ones(img.shape, dtype=np.uint8) * 200
```

<details>
<summary>🔑 Jawaban</summary>

**Perbedaan Krusial:**
- `cv2.add()` melakukan **saturated addition**: jika hasil > 255, dikunci di 255. Contoh: 100 + 200 = 255
- Operator `+` biasa (numpy) melakukan **modular addition**: jika hasil > 255, kembali dari 0 (overflow). Contoh: 100 + 200 = 44 (karena 300 mod 256 = 44)

**Kesimpulan:** Gunakan `cv2.add()` untuk hasil yang "benar" secara visual
</details>

---

**Soal 14.** Jelaskan apa yang dilakukan oleh parameter `alpha` dan `beta` pada kode berikut:

```python
adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
```

<details>
<summary>🔑 Jawaban</summary>

- `alpha` = **kontras**. Nilai `1.0` = normal, `> 1.0` = lebih kontras, `< 1.0` = kurang kontras
- `beta` = **brightness (kecerahan)**. Nilai `0` = normal, `> 0` = lebih terang, `< 0` = lebih gelap
- Formula: `output = alpha * input + beta`
- Pada kode ini: kontras dinaikkan 1.5x dan kecerahan ditambah 30
</details>

---

### ✏️ SOAL TIPE 3: Apa Output dari Kode Berikut?

---

**Soal 15.** Apa output dari kode berikut?

```python
import cv2
img = cv2.imread('foto.jpg')   # foto.jpg berukuran 480x640, berwarna
print(type(img))
print(img.shape)
```

<details>
<summary>🔑 Jawaban</summary>

```
<class 'numpy.ndarray'>
(480, 640, 3)
```
**Penjelasan:**
- `type(img)` → gambar di OpenCV adalah numpy array
- `img.shape` → `(tinggi=480, lebar=640, channel=3)`, 3 channel karena warna BGR
</details>

---

**Soal 16.** Apa output dari kode berikut?

```python
import cv2
img = cv2.imread('foto.jpg')          # gambar 480x640 berwarna
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
print(len(gray.shape))
```

<details>
<summary>🔑 Jawaban</summary>

```
(480, 640)
2
```
**Penjelasan:**
- Setelah konversi ke grayscale, channel hilang → hanya 2 dimensi: `(tinggi, lebar)`
- `len(gray.shape)` = jumlah dimensi = 2
- Bandingkan dengan `img.shape` = `(480, 640, 3)` → `len` = 3
</details>

---

**Soal 17.** Apa output dari kode berikut?

```python
import cv2
img = cv2.imread('foto.jpg')       # gambar 480x640
resized = cv2.resize(img, (224, 224))
print(resized.shape)
```

<details>
<summary>🔑 Jawaban</summary>

```
(224, 224, 3)
```
**Penjelasan:**
- `cv2.resize(img, (lebar, tinggi))` → resize ke 224×224
- Channel (3) tetap karena gambar masih berwarna
- Perhatikan: parameter `resize` = `(lebar, tinggi)`, tapi `shape` = `(tinggi, lebar, channel)`
</details>

---

**Soal 18.** Apa output dari kode berikut?

```python
import cv2
img = cv2.imread('foto.jpg')
pixel = img[100, 200]
print(len(pixel))
print(pixel[0])   # Jika pixel = [120, 80, 50]
```

<details>
<summary>🔑 Jawaban</summary>

```
3
120
```
**Penjelasan:**
- `img[100, 200]` mengambil piksel di baris 100, kolom 200 → mengembalikan array 3 elemen [B, G, R]
- `len(pixel)` = 3 (jumlah channel)
- `pixel[0]` = channel Blue = 120
</details>

---

## 🔷 BAGIAN B — LKM 2: Preprocessing & Face Detection

---

### 📖 Teori Singkat

**Apa itu Haar Cascade?**
Haar Cascade adalah metode **klasik** untuk mendeteksi objek (terutama wajah) dalam gambar. Dikembangkan oleh Paul Viola dan Michael Jones. OpenCV menyediakan file XML yang sudah di-training untuk mendeteksi wajah depan.

**Mengapa harus Grayscale?**
Deteksi wajah Haar Cascade bekerja berdasarkan **perbedaan intensitas** piksel, bukan warna. Jadi input harus grayscale agar lebih cepat dan akurat.

**Apa itu scaleFactor dan minNeighbors?**
- `scaleFactor` = faktor pengecilan gambar di setiap iterasi. Nilai `1.3` artinya gambar diperkecil 30% tiap iterasi. Semakin kecil → lebih teliti tapi lebih lambat
- `minNeighbors` = jumlah tetangga minimum yang harus ada agar area dianggap wajah. Semakin besar → semakin sedikit false positive (lebih ketat)

---

### ✏️ SOAL TIPE 1: Melengkapi Syntax Rumpang

---

**Soal 19.** Lengkapi kode untuk mendeteksi wajah dalam gambar:

```python
import cv2

img = cv2.imread('wajah.jpg')
gray = cv2._____(a)_____(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2._____(b)_____(
    cv2.data.haarcascades + '_____(c)_____'
)

faces = face_cascade._____(d)_____(gray, _____(e)_____, _____(f)_____)

for (x, y, w, h) in faces:
    cv2._____(g)_____(img, (x, y), (_____(h)_____, _____(i)_____), (0, 255, 0), 2)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) cvtColor
(b) CascadeClassifier
(c) haarcascade_frontalface_default.xml
(d) detectMultiScale
(e) 1.3           (scaleFactor)
(f) 5             (minNeighbors)
(g) rectangle
(h) x + w
(i) y + h
```
**Penjelasan:**
- `(x, y)` = sudut kiri atas kotak wajah
- `(x+w, y+h)` = sudut kanan bawah kotak wajah
- `(0, 255, 0)` = warna hijau dalam BGR
- `2` = ketebalan garis kotak
</details>

---

**Soal 20.** Lengkapi kode untuk crop wajah, resize ke 200×200, dan simpan:

```python
for (x, y, w, h) in faces:
    face_crop = _____(a)_____[_____(b)_____:_____(c)_____, _____(d)_____:_____(e)_____]
    face_resized = cv2._____(f)_____(face_crop, (_____(g)_____, _____(h)_____))
    cv2._____(i)_____('wajah_crop.jpg', face_resized)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) gray        (atau img, tergantung mau crop dari grayscale atau berwarna)
(b) y
(c) y + h
(d) x
(e) x + w
(f) resize
(g) 200
(h) 200
(i) imwrite
```
**Penjelasan:**
- Cropping menggunakan **slicing numpy**: `array[baris_awal:baris_akhir, kolom_awal:kolom_akhir]`
- Baris = sumbu Y (vertikal), Kolom = sumbu X (horizontal)
- `y:y+h` = dari baris y sampai y+h (tinggi wajah)
- `x:x+w` = dari kolom x sampai x+w (lebar wajah)
</details>

---

**Soal 21.** Lengkapi kode untuk standarisasi ukuran seluruh gambar dalam dataset:

```python
import cv2
import os

input_dir = 'dataset'
output_dir = 'dataset_resized'

for folder in os._____(a)_____(input_dir):
    folder_path = os.path._____(b)_____(input_dir, folder)
    out_folder = os.path.join(output_dir, folder)
    os._____(c)_____(out_folder, exist_ok=True)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2._____(d)_____(img_path)
        if img is not None:
            resized = cv2._____(e)_____(img, (224, 224))
            cv2._____(f)_____(os.path.join(out_folder, img_name), resized)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) listdir
(b) join
(c) makedirs
(d) imread
(e) resize
(f) imwrite
```
**Penjelasan:**
- `os.listdir()` = menampilkan isi folder
- `os.path.join()` = menggabungkan path (cross-platform)
- `os.makedirs(path, exist_ok=True)` = membuat folder, tidak error jika sudah ada
</details>

---

### ✏️ SOAL TIPE 2: Jelaskan Kode

---

**Soal 22.** Jelaskan fungsi setiap baris:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(f"Jumlah wajah terdeteksi: {len(faces)}")
```

<details>
<summary>🔑 Jawaban</summary>

- **Baris 1-3:** Membuat objek Cascade Classifier dengan memuat file XML model Haar Cascade yang sudah di-training untuk mendeteksi wajah tampak depan. File XML ini adalah bawaan OpenCV (`cv2.data.haarcascades` = path ke folder haarcascades bawaan)
- **Baris 4:** Menjalankan deteksi wajah pada gambar grayscale.
  - `1.3` = scaleFactor (gambar diperkecil 30% tiap iterasi pencarian)
  - `5` = minNeighbors (minimal 5 deteksi tumpang tindih agar dianggap wajah)
  - Hasilnya: list koordinat kotak wajah, masing-masing berisi `(x, y, w, h)`
- **Baris 5:** Menampilkan jumlah wajah yang terdeteksi. `len(faces)` = banyaknya kotak wajah
</details>

---

**Soal 23.** Jelaskan apa arti dari variabel `x, y, w, h` dalam kode berikut:

```python
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

<details>
<summary>🔑 Jawaban</summary>

- `x` = posisi **kolom** (horizontal) sudut kiri atas kotak wajah
- `y` = posisi **baris** (vertikal) sudut kiri atas kotak wajah
- `w` = **lebar** (width) kotak wajah
- `h` = **tinggi** (height) kotak wajah
- `(x, y)` = titik sudut **kiri atas**
- `(x+w, y+h)` = titik sudut **kanan bawah**
- `(0, 255, 0)` = warna garis kotak dalam **BGR** = **hijau**
- `2` = ketebalan garis dalam piksel

```
  (x,y)───────────(x+w,y)
    │                │
    │    WAJAH       │ h (tinggi)
    │                │
  (x,y+h)──────(x+w,y+h)
        w (lebar)
```
</details>

---

### ✏️ SOAL TIPE 3: Prediksi Output

---

**Soal 24.** Apa output dari kode berikut jika gambar berisi 3 wajah?

```python
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(type(faces))
print(len(faces))
print(faces[0])
```

<details>
<summary>🔑 Jawaban</summary>

```
<class 'numpy.ndarray'>
3
[120  85  95 102]
```
**Penjelasan:**
- `type(faces)` = numpy array (kumpulan kotak wajah)
- `len(faces)` = 3 (karena ada 3 wajah)
- `faces[0]` = koordinat wajah pertama `[x, y, w, h]`, nilai angkanya tergantung posisi wajah di gambar
- Jika **tidak ada wajah terdeteksi**, `faces` akan berupa tuple kosong `()`
</details>

---

**Soal 25.** Apa output dari kode berikut?

```python
img = cv2.imread('foto.jpg')         # ukuran 600x800 berwarna
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# Misalkan faces = [(100, 50, 120, 140)]

x, y, w, h = faces[0]
crop = gray[y:y+h, x:x+w]
resized = cv2.resize(crop, (200, 200))

print(crop.shape)
print(resized.shape)
```

<details>
<summary>🔑 Jawaban</summary>

```
(140, 120)
(200, 200)
```
**Penjelasan:**
- `crop = gray[50:190, 100:220]` → crop dari baris 50-190 (tinggi=140) dan kolom 100-220 (lebar=120)
- `crop.shape` = `(140, 120)` → (tinggi, lebar), grayscale jadi tanpa channel
- Setelah `resize(crop, (200, 200))`, menjadi `(200, 200)` → ukuran standar
</details>

---

## 🔷 BAGIAN C — LKM 3: CNN & Transfer Learning

---

### 📖 Teori Singkat

**Apa itu CNN (Convolutional Neural Network)?**
CNN adalah jenis jaringan saraf tiruan yang dirancang khusus untuk memproses data gambar. Arsitekturnya:

```
INPUT → [Conv2D → MaxPooling] × N → Flatten → Dense → OUTPUT
```

**Layer-Layer CNN:**
| Layer | Fungsi | Analogi |
|-------|--------|---------|
| `Conv2D` | Mengekstrak fitur (garis, tepi, tekstur) | Mata yang "memindai" gambar |
| `MaxPooling2D` | Mengecilkan ukuran, menjaga fitur penting | Meng-zoom out untuk melihat gambaran besar |
| `Flatten` | Mengubah data 2D menjadi 1D | Membentangkan peta menjadi daftar |
| `Dense` | Menghubungkan semua neuron (klasifikasi) | Otak yang "memutuskan" ini kelas apa |
| `Dropout` | Mematikan sebagian neuron saat training | Mencegah "menghafal" data training |

**Apa itu Transfer Learning?**
Menggunakan model yang **sudah dilatih** pada dataset besar (misal ImageNet) sebagai dasar, lalu kita **tambahkan layer sendiri** untuk tugas yang lebih spesifik. Analogi: seperti menggunakan skill bahasa Inggris (sudah punya) untuk belajar bahasa Jerman (baru), karena ada kesamaan struktur.

**Activation Function:**
| Fungsi | Kapan Dipakai | Penjelasan |
|--------|--------------|-----------|
| `relu` | Hidden layer | Output = max(0, input). Sederhana & efektif |
| `softmax` | Output layer multi-class (>2 kelas) | Menghasilkan probabilitas, total = 1.0 |
| `sigmoid` | Output layer binary (2 kelas) | Menghasilkan nilai 0–1 |

---

### ✏️ SOAL TIPE 1: Melengkapi Syntax Rumpang

---

**Soal 26.** Lengkapi kode untuk preprocessing dataset dengan ImageDataGenerator:

```python
from tensorflow.keras.preprocessing.image import _____(a)_____

datagen = ImageDataGenerator(
    _____(b)_____ = 1./255,
    validation_split = 0.2
)

train_data = datagen._____(c)_____(
    'dataset/',
    target_size = (_____(d)_____, _____(e)_____),
    batch_size = 32,
    class_mode = '_____(f)_____',
    subset = 'training'
)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) ImageDataGenerator
(b) rescale
(c) flow_from_directory
(d) 224
(e) 224
(f) categorical
```
**Penjelasan:**
- `rescale=1./255` = normalisasi nilai piksel dari 0-255 menjadi 0.0-1.0
- `flow_from_directory` = membaca gambar langsung dari folder, otomatis memberi label sesuai nama subfolder
- `target_size=(224,224)` = resize otomatis ke 224×224
- `class_mode='categorical'` = untuk klasifikasi multi-class (lebih dari 2 kelas)
</details>

---

**Soal 27.** Lengkapi kode untuk membuat model CNN sederhana untuk klasifikasi 5 kelas:

```python
from tensorflow.keras.models import _____(a)_____
from tensorflow.keras.layers import _____(b)_____, MaxPooling2D, _____(c)_____, Dense

model = Sequential([
    _____(d)_____(32, (3, 3), activation='_____(e)_____', input_shape=(224, 224, 3)),
    _____(f)_____(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    _____(g)_____(),
    Dense(128, activation='relu'),
    Dense(_____(h)_____, activation='_____(i)_____')
])
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) Sequential
(b) Conv2D
(c) Flatten
(d) Conv2D
(e) relu
(f) MaxPooling2D
(g) Flatten
(h) 5
(i) softmax
```
**Penjelasan:**
- `Sequential` = model yang layer-nya disusun berurutan
- `Conv2D(32, (3,3))` = 32 filter konvolusi ukuran 3×3
- `activation='relu'` = fungsi aktivasi ReLU untuk hidden layer
- `input_shape=(224,224,3)` = **hanya di layer pertama**, menentukan ukuran input
- `Flatten()` = mengubah output 2D (matriks) menjadi 1D (vektor)
- `Dense(5, activation='softmax')` = output 5 kelas dengan probabilitas (total = 1.0)
</details>

---

**Soal 28.** Lengkapi kode untuk compile dan training model:

```python
model._____(a)_____(
    optimizer = '_____(b)_____',
    loss = '_____(c)_____',
    metrics = ['_____(d)_____']
)

history = model._____(e)_____(
    train_data,
    _____(f)_____ = 10,
    validation_data = val_data
)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) compile
(b) adam
(c) categorical_crossentropy
(d) accuracy
(e) fit
(f) epochs
```
**Penjelasan:**
- `compile` = mengkonfigurasi model sebelum training
- `adam` = optimizer adaptif yang paling sering dipakai
- `categorical_crossentropy` = loss function untuk multi-class
- `accuracy` = metrik yang ingin kita pantau
- `fit` = memulai proses training
- `epochs=10` = data training diproses 10 kali putaran penuh
</details>

---

**Soal 29.** Lengkapi kode Transfer Learning dengan MobileNetV2:

```python
from tensorflow.keras.applications import _____(a)_____
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

base_model = MobileNetV2(
    weights = '_____(b)_____',
    _____(c)_____ = False,
    input_shape = (224, 224, 3)
)

base_model._____(d)_____ = False

x = base_model._____(e)_____
x = _____(f)_____()(x)
x = Dense(128, activation='relu')(x)
output = Dense(5, activation='softmax')(x)

model_tl = _____(g)_____(inputs=base_model._____(h)_____, outputs=output)
```

<details>
<summary>🔑 Jawaban</summary>

```
(a) MobileNetV2
(b) imagenet
(c) include_top
(d) trainable
(e) output
(f) GlobalAveragePooling2D
(g) Model
(h) input
```
**Penjelasan:**
- `weights='imagenet'` = menggunakan bobot yang sudah dilatih pada dataset ImageNet (1.4 juta gambar, 1000 kelas)
- `include_top=False` = **tidak** menyertakan layer klasifikasi asli (1000 kelas ImageNet)
- `base_model.trainable = False` = **membekukan** (freeze) semua bobot MobileNetV2 agar tidak berubah saat training
- `GlobalAveragePooling2D` = mengambil rata-rata dari setiap feature map → mengecilkan dimensi
- `Model(inputs=..., outputs=...)` = membuat model baru yang menggabungkan base_model + layer custom
</details>

---

### ✏️ SOAL TIPE 2: Jelaskan Kode

---

**Soal 30.** Jelaskan perbedaan antara model CNN Baseline dan Transfer Learning:

```python
# Model A (CNN Baseline)
model_a = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(5, activation='softmax')
])

# Model B (Transfer Learning)
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
output = Dense(5, activation='softmax')(x)
model_b = Model(inputs=base.input, outputs=output)
```

<details>
<summary>🔑 Jawaban</summary>

| Aspek | Model A (Baseline) | Model B (Transfer Learning) |
|-------|--------------------|-----------------------------|
| Bobot awal | **Random** (acak) | **Pre-trained** (sudah terlatih dari ImageNet) |
| Perlu data training | **Banyak** | **Sedikit** sudah cukup |
| Waktu training | **Lama** | **Cepat** (hanya train layer atas) |
| Akurasi | **Rendah** (terutama jika data sedikit) | **Tinggi** (memanfaatkan pengetahuan sebelumnya) |
| Kapan dipakai | Dataset besar, tugas unik | Dataset kecil, tugas mirip ImageNet |

- Model A belajar **dari nol** — semua bobot diinisialisasi acak
- Model B menggunakan **pengetahuan yang sudah ada** dari MobileNetV2 yang sudah dilatih mengenali 1000 objek, lalu hanya melatih layer terakhir untuk tugas kita (5 kelas wajah)
</details>

---

**Soal 31.** Jelaskan apa arti `include_top=False` dan `base_model.trainable = False`:

<details>
<summary>🔑 Jawaban</summary>

**`include_top=False`:**
- "Top" = layer **fully connected teratas** pada model asli
- MobileNetV2 asli didesain untuk 1000 kelas ImageNet
- `include_top=False` artinya kita **membuang** layer klasifikasi 1000 kelas tersebut
- Kita hanya mengambil **bagian feature extractor** (Conv layers) nya saja
- Lalu kita tambah layer Dense sendiri sesuai kebutuhan (misal 5 kelas)

**`base_model.trainable = False`:**
- Artinya semua bobot (weight) pada MobileNetV2 **dibekukan / di-freeze**
- Saat training, bobot tersebut **TIDAK akan berubah**
- Hanya layer yang kita tambahkan sendiri (Dense, GlobalAveragePooling) yang bobotnya di-update
- Ini membuat training **jauh lebih cepat** dan mencegah overfitting
</details>

---

### ✏️ SOAL TIPE 3: Prediksi Output

---

**Soal 32.** Apa output dari kode berikut?

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Layer terakhir: Dense(5, activation='softmax')
```

Jika model diberi input gambar ukuran 224×224×3, berapa dimensi output layer terakhir?

<details>
<summary>🔑 Jawaban</summary>

```
Output shape layer terakhir: (None, 5)
```
**Penjelasan:**
- `Dense(5, activation='softmax')` menghasilkan **5 nilai probabilitas**
- Contoh output: `[0.05, 0.10, 0.70, 0.10, 0.05]` → total = 1.0
- Kelas dengan probabilitas tertinggi = prediksi model (kelas ke-3 = 0.70)
- `None` pada shape berarti batch size (jumlah gambar yang diproses sekaligus)
</details>

---

**Soal 33.** Apa output dari kode berikut?

```python
history = model.fit(train_data, epochs=3, validation_data=val_data)
print(list(history.history.keys()))
```

<details>
<summary>🔑 Jawaban</summary>

```
['loss', 'accuracy', 'val_loss', 'val_accuracy']
```
**Penjelasan:**
- `history.history` = dictionary berisi riwayat metrik selama training
- `loss` = nilai loss pada data training
- `accuracy` = akurasi pada data training
- `val_loss` = nilai loss pada data validasi
- `val_accuracy` = akurasi pada data validasi
- Setiap key berisi **list** sepanjang jumlah epoch (3 nilai)
</details>

---

**Soal 34.** Apa output dari kode berikut?

```python
loss, accuracy = model.evaluate(val_data)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
```

Jika loss = 0.3521 dan accuracy = 0.8750, apa outputnya?

<details>
<summary>🔑 Jawaban</summary>

```
Loss: 0.3521
Accuracy: 0.8750
```
**Penjelasan:**
- `model.evaluate()` mengembalikan **2 nilai**: loss dan accuracy
- `:.4f` = format angka desimal 4 digit di belakang koma
- Loss 0.35 = cukup baik (semakin kecil semakin baik)
- Accuracy 0.875 = 87.5% benar (semakin besar semakin baik)
</details>

---

## 🔷 BAGIAN D — Soal Campuran (Simulasi Ujian)

---

**Soal 35.** Tulis kode **lengkap** untuk:
1. Membaca gambar `kelas.jpg`
2. Konversi ke grayscale
3. Deteksi semua wajah
4. Crop setiap wajah, resize ke 200×200
5. Simpan setiap wajah ke file `wajah_1.jpg`, `wajah_2.jpg`, dst.

<details>
<summary>🔑 Jawaban</summary>

```python
import cv2

# 1. Baca gambar
img = cv2.imread('kelas.jpg')

# 2. Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 4 & 5. Crop, resize, dan simpan
for i, (x, y, w, h) in enumerate(faces):
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (200, 200))
    cv2.imwrite(f'wajah_{i+1}.jpg', face_resized)

print(f"Total wajah tersimpan: {len(faces)}")
```
</details>

---

**Soal 36.** Tulis kode **lengkap** untuk membuat model CNN dengan arsitektur:
- Conv2D 32 filter → MaxPooling → Conv2D 64 filter → MaxPooling → Conv2D 128 filter → MaxPooling → Flatten → Dense 256 → Dropout 0.5 → Dense 10 (output)
- Compile dengan optimizer adam
- Training 15 epoch

<details>
<summary>🔑 Jawaban</summary>

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_data, epochs=15, validation_data=val_data)
```
</details>

---

**Soal 37.** Tulis kode **lengkap** untuk implementasi Transfer Learning menggunakan MobileNetV2 untuk klasifikasi 3 kelas:

<details>
<summary>🔑 Jawaban</summary>

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Load pre-trained model tanpa top layer
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Tambah layer custom
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

# Buat model
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(train_data, epochs=10, validation_data=val_data)

# Evaluasi
loss, acc = model.evaluate(val_data)
print(f"Accuracy: {acc:.4f}")
```
</details>

---

## 📌 Rangkuman Akhir — Yang WAJIB Dihafal

### Pola 1: Deteksi Wajah (Paling Penting!)
```
imread → cvtColor(BGR2GRAY) → CascadeClassifier → detectMultiScale → rectangle
```

### Pola 2: Crop Wajah
```
gray[y:y+h, x:x+w] → resize(crop, (200,200)) → imwrite
```

### Pola 3: CNN
```
Conv2D → MaxPooling2D → Flatten → Dense → compile → fit
```

### Pola 4: Transfer Learning
```
MobileNetV2(include_top=False) → trainable=False → GlobalAveragePooling2D → Dense → Model → compile → fit
```

> [!CAUTION]
> **LATIHAN = KUNCI!** Cetak soal-soal di atas, tutup jawaban, dan kerjakan di kertas kosong. Ulangi sampai bisa tanpa melihat. Semangat UTS! 💪
