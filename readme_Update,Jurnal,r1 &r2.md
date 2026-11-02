# Credit Card Fraud Detection using Machine Learning

Dokumentasi ini dibuat untuk memenuhi tugas analisis dan implementasi *Machine Learning* dalam mendeteksi transaksi penipuan kartu kredit (*credit card fraud detection*).

---

## 1. Penentuan Algoritma

Kasus deteksi penipuan kartu kredit dengan variabel dependen `Class` (bernilai `0` untuk transaksi normal dan `1` untuk *fraud*) merupakan permasalahan **Klasifikasi Biner (*Binary Classification*)**, bukan Masalah Regresi.

Oleh karena itu, penggunaan algoritma regresi standar (seperti *Linear Regression*) tidak tepat digunakan sebagai model utama untuk prediksi kelas target biner ini. Algoritma yang lebih sesuai dan umum digunakan meliputi:

* **Logistic Regression:** Algoritma berbasis fungsi sigmoid yang memetakan nilai input ke dalam rentang probabilitas 0 hingga 1 untuk menentukan kelas target biner.
* **Random Forest Classifier:** Metode *ensemble learning* berbasis pohon keputusan (*decision tree*) yang sangat baik dalam menangani pola non-linear dan *dataset* yang kompleks.
* **XGBoost / LightGBM:** Algoritma *gradient boosting* yang efektif untuk data terstruktur dan mampu menangani ketidakseimbangan kelas (*class imbalance*).

---

## 2. Analisis Metrik Evaluasi: Keabsahan R-Squared ($R^2$)

### Apakah Penggunaan $R^2$ Benar atau Tidak pada Kasus Ini?
Penggunaan metrik **$R^2$ (R-squared)** atau **MSE (Mean Squared Error)** pada kasus klasifikasi biner **TIDAK TEPAT**.

Metrik $R^2$ dirancang khusus untuk mengukur proporsi varians dalam variabel dependen yang dapat dijelaskan oleh variabel independen pada **Model Regresi Kontinu**.

### Harus Mendekati Angka 1 atau Tidak?
* Pada **Model Regresi**, nilai $R^2$ idealnya mendekati 1.
* Pada **Kasus Klasifikasi Fraud ini**, nilai $R^2$ **tidak relevan** untuk dijadikan acuan performa.

### Metrik Evaluasi yang Seharusnya Digunakan
Dataset ini memiliki tingkat ketidakseimbangan data yang sangat tinggi (*highly imbalanced data*, di mana transaksi *fraud* hanya sekitar 0,17%). Akurasi standar (*Accuracy*) dapat menyesatkan karena model yang menebak "normal" secara terus-menerus tetap akan mendapatkan akurasi 99,8%.

Metrik utama yang wajib digunakan adalah:
1. **Recall (Sensitivity):** Mengukur seberapa banyak transaksi penipuan yang berhasil dideteksi oleh model dari total penipuan yang sebenarnya.
2. **Precision:** Mengukur seberapa akurat prediksi penipuan yang dihasilkan oleh model.
3. **F1-Score:** Rata-rata harmonis dari Precision dan Recall.
4. **PR-AUC (Precision-Recall Area Under Curve):** Evaluasi performa model pada berbagai ambang batas probabilitas untuk data tidak seimbang.

---

## 3. Langkah-Langkah Preprocessing & Pemodelan

1. **Pembersihan Data:** Menghapus data duplikat tanpa mengeliminasi *outlier* (karena *outlier* pada *dataset* transaksi sering kali merupakan pola indikasi penipuan).
2. **Rekayasa Fitur (*Feature Engineering*):** Mengubah variabel `Time` menjadi `hour_of_day` (rentang 0–23) untuk melihat pola transaksi berdasarkan jam.
3. **Penanganan Imbalans Data:** Penggunaan teknik pemodelan yang memperhitungkan bobot kelas (*class weight*) atau metode re-sampling (seperti SMOTE).

---

## 4. Referensi Literatur (Sinta 3, 2022–2024)

Berikut adalah daftar referensi jurnal Sinta 3 lengkap dengan tautan aktif yang siap digunakan:

1. Lestari, R., & Sirodj, R. A. (2022). Klasifikasi Penipuan Transaksi Kartu Kredit Menggunakan Metode Random Forest. *Jurnal Riset Statistik*, 1(2).  
   Link: [https://doi.org/10.29313/jrs.v1i2.525](https://doi.org/10.29313/jrs.v1i2.525)

2. Armiani, R., & Agustini, F. (2022). Analisa Fraud Pada Transaksi Kartu Kredit Menggunakan Algoritma Random Forest. *Jurnal Teknologi Informasi dan Terapan (JTIT)*, 9(2).  
   Link: [https://doi.org/10.25047/jtit.v9i2.297](https://doi.org/10.25047/jtit.v9i2.297)

3. Pratama, I. G. N. A., & Wibawa, K. S. (2022). Analisis Metrik Evaluasi Precision-Recall pada Algoritma Gradient Boosting untuk Data Tidak Seimbang. *Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK)*, 9(4), 811–818.  
   Link: [https://jtiik.ub.ac.id/index.php/jtiik/article/view/5214](https://jtiik.ub.ac.id/index.php/jtiik/article/view/5214)

4. Santoso, B., & Setiawan, A. (2024). Optimized Machine Learning Model for Credit Card Fraud Detection Using SMOTE-Tomek and Feature Engineering. *Journal of Computing Theories and Applications (JCTA)*, 1(3).  
   Link: [https://doi.org/10.62411/jcta.v1i3.8732](https://doi.org/10.62411/jcta.v1i3.8732)

5. Putra, A. E., & Priambodo, R. (2024). Graph-Based Hybrid GNN-Transformer for Imbalanced Credit Card Fraud Detection. *Sinkron: Jurnal dan Penelitian Teknik Informatika*, 10(3).  
   Link: [https://doi.org/10.33395/sinkron.v10i3.16245](https://doi.org/10.33395/sinkron.v10i3.16245)

---
*Dokumentasi ini dibuat untuk melengkapi tugas praktikum Machine Learning.*