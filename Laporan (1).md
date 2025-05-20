# Laporan Proyek Machine Learning - Agus Irvan Maulana

## Project Overview
Sistem rekomendasi buku bertujuan untuk membantu pengguna menemukan buku yang relevan dan menarik berdasarkan preferensi mereka. Proyek ini menggabungkan dua pendekatan utama :
- Content-Based Filtering untuk merekomendasikan buku dengan karakteristik serupa dengan yang disukai pengguna.
- Collaborative Filtering untuk menyarankan buku berdasarkan perilaku pengguna lain yang mirip.

Pada bagian ini, Kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.


## Business Understanding
### Problem Statements
Pengguna sering kali kesulitan menemukan buku yang sesuai dengan minat dan kebutuhannya karena banyaknya pilihan yang tersedia. Hal ini menyebabkan :
- Rendahnya keterlibatan pengguna
- Menurunnya kepuasan pengguna dalam menggunakan platform
- Kurangnya personalisasi dalam pengalaman membaca
  
### Goals
- Memberikan rekomendasi buku yang relevan dan dipersonalisasi untuk setiap pengguna
- Meningkatkan user engagement dan waktu yang dihabiskan dalam platform
- Meningkatkan konversi terhadap pembelian atau pembacaan buku



### Solution statements
#### Content-Based Filtering
- Menggunakan metadata buku seperti judul, penulis, kategori, bahasa, dan genre
- Menganalisis kesamaan antara buku berdasarkan deskripsi kontennya (TF-IDF, cosine similarity, dll.)
- Memberikan rekomendasi buku yang mirip dengan yang sudah disukai oleh pengguna

#### Content-Based Filtering
- Menggunakan data interaksi pengguna (user_id, isbn, dan rating)
- Membangun model dengan pendekatan matrix factorization (misalnya: Embedding layer pada TensorFlow atau algoritma seperti SVD)
- Merekomendasikan buku berdasarkan preferensi pengguna lain yang mirip
  
## Data Understanding
Dataset yang digunakan berisi 1.031.175 entri yang mencatat interaksi pengguna terhadap buku.data ini berasal dari [kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset?select=Books+Data+with+Category+Language+and+Summary) .

Variabel-variabel pada dataset adalah sebagai berikut:
- Unnamed: 0 : Index bawaan dari proses penyimpanan/penggabungan data
- user_id : ID unik pengguna. Digunakan untuk mengidentifikasi siapa yang membaca atau memberi rating pada buku.
- location : Informasi lokasi pengguna dalam format gabungan (biasanya City, State, Country)
- age : usia pengguna
- isbn : Nomor identifikasi unik untuk setiap buku (International Standard Book Number)
- rating : Nilai rating yang diberikan oleh pengguna terhadap buku
- book_title : Judul  dari buku
- book_author : Nama penulis buku
- year_of_publication : Tahun terbit buku
- Nama penerbit buku
- img_s/img_m/img_l : URL gambar sampul buku ukuran kecil, sedang, besar
- Summary : Ringkasan/deskripsi buku
- Language : Bahasa buku
- Category : Kategori/genre buku
- city : Nama kota penggun
- state : Nama negara bagian pengguna
- country : Negara pengguna

#### Jumlah missing value pada dataset :
![image](https://github.com/user-attachments/assets/aa7e3e58-d3fc-4390-8078-8649780aa935)

Dataset memiliki missing value pada kolom book_author, city, state, dan country, dengan jumlah terbanyak pada country yaitu 35.374 data yang hilang.

#### Jumlah Duplikat Data
![image](https://github.com/user-attachments/assets/ceab0796-daef-4ed2-aa97-d5ec993a693c)

Dataset tidak memiliki data duplikat, sehingga seluruh baris bersifat unik.

## Data Preparation
### 1.menghapus missing value
Menghapus semua baris yang memiliki nilai kosong (missing value) agar tidak mengganggu proses modeling, terutama dalam kolom penting seperti book_author, city, state, dan country
### 2. Menghapus Rating 0
Baris dengan rating = 0 dihapus karena dianggap tidak memberikan evaluasi terhadap buku
### 3.Mapping ID ke Bentuk Numerik (Encoding)
- Melakukan encoding user dan ISBN ke angka untuk keperluan input ke dalam model 
- Membuat dictionary encode_user_id dan encode_book_id untuk mapping dua arah (ke dan dari ID asli)
### 4. Normalisasi Rating 
Melakukan normalisasi nilai rating ke rentang 0–1 agar sesuai dengan fungsi aktivasi sigmoid dalam model Collaborative Filtering
### 5. Split Dataset
Data dibagi menjadi 80% data latih dan 20% data validasi untuk mengevaluasi kinerja model.
### 6.TF-IDF Vectorization
Mengubah kategori buku menjadi representasi vektor menggunakan TF-IDF, lalu menghitung cosine similarity antar buku untuk Content-Based Filtering


## Modeling
### Content Based Filtering
Content-Based Filtering adalah salah satu teknik dalam sistem rekomendasi yang menggunakan informasi karakteristik konten dari setiap item untuk memberikan rekomendasi. Sistem ini menganalisis fitur-fitur dari item yang sudah disukai oleh pengguna, lalu mencari item lain yang memiliki fitur serupa untuk direkomendasikan.
#### Kelebihan 
- Rekomendasi disesuaikan dengan preferensi unik pengguna berdasarkan item yang sudah mereka sukai sebelumnya
- Sistem bisa bekerja tanpa memerlukan data dari pengguna lain
- Bisa merekomendasikan item baru selama fitur item tersebut tersedia

#### Kekurangan  
- Cenderung merekomendasikan item yang sangat mirip dengan yang sudah diketahui pengguna
- Membutuhkan representasi fitur yang lengkap dan berkualitas tinggi agar rekomendasi akurat
- Bisa merekomendasikan item baru selama fitur item tersebut tersedia
- Kadang-kadang sulit menangkap preferensi pengguna yang kompleks

### Colaborative Filtering
Collaborative Filtering adalah teknik dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan preferensi atau perilaku pengguna lain yang memiliki kesamaan dengan pengguna target. Sistem ini tidak bergantung pada konten item, melainkan pada pola interaksi pengguna dengan item 
#### Kelebihan 
- Dapat memberikan rekomendasi yang beragam dan tidak terbatas hanya pada fitur item tertentu
- Mampu menangkap preferensi pengguna yang kompleks melalui pola interaksi komunitas pengguna

#### Kekurangan  
- Membutuhkan data pengguna dan interaksi yang cukup banyak agar rekomendasi bisa akurat 
- Rentan terhadap masalah sparsity, yaitu data interaksi yang sangat sedikit sehingga sulit menemukan kemiripan antar pengguna atau item
- Bisa terjadi bias popularitas, dimana item yang populer lebih sering direkomendasikan sehingga item niche kurang mendapat perhatian
  
## Evaluation
### Content Based Filtering
Precision adalah salah satu metrik evaluasi yang sering digunakan untuk menilai kualitas sistem rekomendasi, termasuk Content-Based Filtering. Precision mengukur seberapa tepat rekomendasi yang diberikan oleh sistem.

Precision = (Jumlah rekomendasi buku yang relevan) / (Jumlah item yang direkomendasikan)

![image](https://github.com/user-attachments/assets/f6c09a38-aa49-4c49-a305-c0174725dd41)

Berdasarkan hasil rekomendasi, semua dari 10 buku yang ditampilkan relevan dengan kategori yang sesuai, sehingga nilai precision mencapai 1.0 atau 100%
    
### Colaborative Filtering
Root Mean Square Error (RMSE) digunakan untuk mengukur seberapa dekat prediksi sistem rekomendasi dengan rating sebenarnya yang diberikan oleh pengguna. RMSE menghitung rata-rata kuadrat dari selisih antara rating yang diprediksi dan rating aktual, lalu diakarkan.

![image](https://github.com/user-attachments/assets/03b8541f-eab8-4660-a45d-a91cc537cc48)



Model telah dilatih selama 50 epoch dengan penurunan nilai loss dan root mean squared error (RMSE) yang konsisten pada data pelatihan maupun validasi. Pada awal pelatihan (epoch 1), nilai val_root_mean_squared_error berada di angka 0.3063, dan secara bertahap mengalami perbaikan hingga mencapai 0.2280 pada epoch ke-35 serta terus menurun hingga 0.2276 pada epoch ke-50. Tren penurunan ini menunjukkan bahwa model mampu meningkatkan kemampuan generalisasi terhadap data validasi seiring berjalannya pelatihan.
