# Machine-Learning-Olah-Data-Cek-Penipuan-pada-Kartu-Kredit-
Kode jupyter notebook saya yang digunakan untuk olah data Penipuan pada kartu kredit 


# Deteksi Penipuan Kartu Kredit (Credit Card Fraud Detection)

Tugas Machine Learning: membangun model klasifikasi untuk mendeteksi transaksi kartu kredit yang **fraud (penipuan)** dan **normal**.

## Dataset

*KARENA KETERBATASAN UKURAN FILE, DATASET YG SAYA CANTUMKAN DI REPOSITORY ITU HASIL KOMPRESS, DIBAWAH INI ADALAH LINK DATASET VERSI LENGKAPNYA:
- **Sumber:** [Credit Card Fraud Detection (Kaggle)] https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Dataset lengkap:** 284.807 transaksi, 31 kolom, hanya 492 transaksi fraud (±0,17%)
- **Kolom:**
  - `Time`: detik sejak transaksi pertama
  - `V1`–`V28`: fitur hasil PCA (sudah dianonimkan)
  - `Amount`: nominal transaksi
  - `Class`: target (0 = normal, 1 = fraud)

> **Catatan:** Karena batas ukuran file GitHub, file `creditcard.csv` di repo ini adalah **sampel** (43.492 baris: seluruh 492 data fraud + 43.000 data normal yang dipilih acak). Seluruh hasil dan output di notebook berasal dari **dataset lengkap**. Jika notebook dijalankan ulang dengan file sampel, angka hasilnya akan berbeda karena proporsi fraud lebih besar. Dataset lengkap dapat diunduh dari link Kaggle di atas.

## Tahapan

1. **Loading**: membaca data `creditcard.csv`
2. **Cleaning & Transformation**
   - Tidak ada missing value
   - 1.081 baris duplikat dihapus (menjadi 283.726 baris)
   - Outlier tidak dihapus karena transaksi fraud sering berupa nilai ekstrem
   - `Time` diubah menjadi `hour_of_day` (jam dalam sehari)
3. **EDA (Exploratory Data Analysis)**: distribusi kelas, distribusi `Amount`, pola jam transaksi, korelasi fitur terhadap `Class`, dan box plot fitur teratas
4. **ExDA (Explanatory Data Analysis)**: ringkasan temuan utama beserta visualisasi dan kesimpulan
5. **Splitting**: 80% data latih dan 20% data uji dengan `stratify` agar proporsi fraud sama di kedua bagian. Scaling (`StandardScaler`) hanya pada `Amount` dan `hour_of_day`, di-fit pada data latih saja
6. **Modeling**: Logistic Regression, Decision Tree, Random Forest, dan Gradient Boosting (`HistGradientBoostingClassifier`) dengan `class_weight='balanced'` untuk menangani data yang tidak seimbang
7. **Deployment & Monitoring**
   - Model dan preprocessing disimpan dengan `joblib`/`pickle`
   - Fungsi prediksi untuk transaksi baru dan contoh API (FastAPI)
   - Logging prediksi, deteksi data drift dengan PSI, dan alarm performa per batch

## Hasil (pada data uji, kelas Fraud)

Karena data sangat tidak seimbang, accuracy tidak dipakai sebagai patokan utama. Metrik utama adalah Precision, Recall, F1, dan PR-AUC.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,0630 | 0,8737 | 0,1175 | 0,9670 | 0,6772 |
| Decision Tree | 0,0773 | 0,7684 | 0,1405 | 0,8776 | 0,5018 |
| **Random Forest** | **0,8452** | 0,7474 | **0,7933** | **0,9721** | **0,7755** |
| Gradient Boosting | 0,3319 | 0,8000 | 0,4691 | 0,9264 | 0,6821 |

**Model terbaik: Random Forest** (dipilih berdasarkan PR-AUC), dengan Precision 0,845 dan Recall 0,747 pada kelas fraud.

## Cara Menjalankan

1. Clone repo ini
2. Install library:
```
   pip install pandas numpy seaborn matplotlib scikit-learn joblib
```
   (Opsional untuk API: `pip install fastapi uvicorn`)
3. Pastikan file `creditcard.csv` berada satu folder dengan notebook
4. Jalankan notebook `MachineLearn_DataCreditCard_fixed.ipynb` dari atas sampai bawah

## Struktur File

```
├── MachineLearn_DataCreditCard_fixed.ipynb   # notebook utama
├── creditcard.csv                            # dataset (sampel)
└── README.md
```

File yang dihasilkan saat notebook dijalankan: `fraud_model.joblib`, `fraud_model.pkl`, `app.py`, `prediction_log.csv`.


## Credit Card Fraud Detection using Machine Learning

Dokumentasi ini dibuat untuk memenuhi tugas analisis dan implementasi Machine Learning dalam mendeteksi transaksi penipuan kartu kredit (fraud detection).

1. Penentuan Algoritma

Kasus deteksi penipuan kartu kredit dengan variabel dependen Class (bernilai 0 untuk transaksi normal dan 1 untuk fraud) merupakan permasalahan Klasifikasi Biner (Binary Classification), bukan Masalah Regresi.

Oleh karena itu, penggunaan algoritma regresi standar (seperti Linear Regression) tidak tepat digunakan sebagai model utama untuk prediksi kelas target biner ini. Algoritma yang lebih sesuai dan umum digunakan meliputi:

- Logistic Regression: Algoritma berbasis fungsi sigmoid yang memetakan nilai input ke dalam rentang probabilitas 0 hingga 1 untuk menentukan kelas target biner.

- Random Forest Classifier: Metode ensemble learning berbasis pohon keputusan (decision tree) yang sangat baik dalam menangani pola non-linear dan dataset yang kompleks.

- XGBoost / LightGBM: Algoritma gradient boosting yang efektif untuk data terstruktur dan mampu menangani ketidakseimbangan kelas (class imbalance).

2. Analisis Metrik Evaluasi: Keabsahan R-Squared ($R^2$)

Apakah Penggunaan $R^2$ Benar atau Tidak pada Kasus Ini?
Penggunaan metrik $R^2$ (R-squared) atau MSE (Mean Squared Error) pada kasus klasifikasi biner TIDAK TEPAT.
Metrik $R^2$ dirancang khusus untuk mengukur proporsi varians dalam variabel dependen yang dapat dijelaskan oleh variabel independen pada Model Regresi Kontinu.

Harus Mendekati Angka 1 atau Tidak?

Pada Model Regresi, nilai $R^2$ idealnya mendekati 1.
Pada Kasus Klasifikasi Fraud ini, nilai $R^2$ tidak relevan untuk dijadikan acuan performa.
Metrik Evaluasi yang Seharusnya Digunakan

Dataset ini memiliki tingkat ketidakseimbangan data yang sangat tinggi (highly imbalanced data, di mana transaksi fraud hanya sekitar 0,17%). Akurasi standar (Accuracy) juga dapat menyesatkan karena model yang menebak "normal" secara terus-menerus tetap akan mendapatkan akurasi 99,8%.

Metrik utama yang wajib digunakan adalah:

- Recall (Sensitivity): Mengukur seberapa banyak transaksi penipuan yang berhasil dideteksi oleh model dari total penipuan yang sebenarnya.
-  Precision: Mengukur seberapa akurat prediksi penipuan yang dihasilkan oleh model.
-  F1-Score: Rata-rata harmonis dari Precision dan Recall.
-  PR-AUC (Precision-Recall Area Under Curve): Evaluasi performa model pada berbagai ambang batas probabilitas untuk data tidak seimbang.

3. Langkah-Langkah Preprocessing & Pemodelan (Berdasarkan Notebook)

-Pembersihan Data: Menghapus data duplikat tanpa mengeliminasi outlier (karena outlier pada dataset transaksi sering kali merupakan pola indikasi penipuan).

-Rekayasa Fitur (Feature Engineering): Mengubah variabel Time menjadi hour_of_day (rentang 0–23) untuk melihat pola transaksi berdasarkan jam.

-Penanganan Imbalans Data: Penggunaan teknik pemodelan yang memperhitungkan bobot kelas (class weight) atau metode re-sampling (seperti SMOTE).

4. Referensi Literatur Ber-link (Sinta 3, 2022–2024)

Berikut adalah daftar referensi jurnal Sinta 3 lengkap dengan tautannya. Anda dapat langsung meng-copy seluruh teks referensi di bawah ini melalui kotak kode:

1. Lestari, R., & Sirodj, R. A. (2022). Klasifikasi Penipuan Transaksi Kartu Kredit Menggunakan Metode Random Forest. Jurnal Riset Statistik, 1(2). 
Link: https://doi.org/10.29313/jrs.v1i2.525

2. Armiani, R., & Agustini, F. (2022). Analisa Fraud Pada Transaksi Kartu Kredit Menggunakan Algoritma Random Forest. Jurnal Teknologi Informasi dan Terapan (JTIT), 9(2). 
Link: https://doi.org/10.25047/jtit.v9i2.297

3. Pratama, I. G. N. A., & Wibawa, K. S. (2022). Analisis Metrik Evaluasi Precision-Recall pada Algoritma Gradient Boosting untuk Data Tidak Seimbang. Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK), 9(4), 811–818. 
Link: https://jtiik.ub.ac.id/index.php/jtiik/article/view/5214

4. Santoso, B., & Setiawan, A. (2024). Optimized Machine Learning Model for Credit Card Fraud Detection Using SMOTE-Tomek and Feature Engineering. Journal of Computing Theories and Applications (JCTA), 1(3). 
Link: https://doi.org/10.62411/jcta.v1i3.8732

5. Putra, A. E., & Priambodo, R. (2024). Graph-Based Hybrid GNN-Transformer for Imbalanced Credit Card Fraud Detection. Sinkron: Jurnal dan Penelitian Teknik Informatika, 10(3). 
Link: https://doi.org/10.33395/sinkron.v10i3.16245

