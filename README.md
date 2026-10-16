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
