# Laporan Project Machine Learning
## Employee Attrition Prediction Using ... Classification Modelling

**Hammam Maulana Arijudin - 5003231013**

**Bagas Rakhawisesa - 5003231133**

**Faiz Nabilianto - 5003231143**

---
## Daftar Isi
- [Project Domain: Employment](#project-domain-employment)
  - [References](#references)
- [Business Understanding](#business-understanding)
  - [Problem Statements](#problem-statements)
  - [Goals](#goals)
  - [Solution Statements](#solution-statements)
  - [Project Benefits](#project-benefits)
- [Data Understanding](#data-understanding)
  - [Data Source](#data-source)
  - [Feature Description](#feature-description)
  - [Exploratory Data Analysis (EDA)](#explaratory-data-analysis---deskripsi-variabel)
- [Data Preparation](#data-preparation)
  - [Label Encoding](#label-encoding)
  - [Drop Features](#drop-features)
  - [Transformation](#transformation)
  - [Scaling and Encoding](#scaling-annd-encoding)
  - [Feature Selection](#features-selection)
  - [Splitting Dataset](#splitting-dataset)
  - [Handling Imbalanced Dataset](#handling-imbalanced-dataset)
- [Modelling with Support Vector Machine](#modelling-with-support-vector-machine)
  - [SVM](#svm)
  - [Modelling](#modelling)
  - [Model Calbiration and Evaluation](#model-calibration-and-evaluation)

  
---
## **Project Domain: Employment**
Analisis employee attrition merupakan salah satu aspek penting dalam Human Resource (HR) Analytics, yaitu penerapan analisis data untuk memahami, mengukur, dan meningkatkan efektivitas sumber daya manusia di organisasi. [[1](https://www.talenta.co/blog/pengertian-hr-analytics/)]
Dalam konteks bisnis modern, kehilangan karyawan (attrition) tidak hanya berdampak pada berkurangnya tenaga kerja, tetapi juga menimbulkan biaya signifikan yang mencakup proses rekrutmen, pelatihan, hingga hilangnya produktivitas serta pengetahuan institusional. [[2](https://www.hrbench.com/resource/learn/cost-of-turnover)]
Oleh karena itu, memahami penyebab dan pola dari attrition menjadi langkah strategis bagi organisasi dalam menjaga stabilitas tenaga kerja dan efisiensi operasional.

Faktor-faktor yang memengaruhi attrition dapat mencakup kepuasan kerja, beban kerja, peluang karier, tingkat kompensasi, hingga hubungan dengan manajer dan rekan kerja [[3](https://www.researchgate.net/publication/362296472)]
Dengan memahami variabel-variabel ini secara kuantitatif, organisasi dapat mengidentifikasi karyawan dengan risiko tinggi untuk keluar dan merancang strategi retensi yang lebih efektif.


### **References**

[1] Talenta (2023). *Pengertian HR Analytics: Fungsi, Manfaat, dan Contohnya*. Retrieved from [https://www.talenta.co](https://www.talenta.co/blog/pengertian-hr-analytics/)

[2] HRBench (2024). *The Cost of Turnover*. Retrieved from [https://www.hrbench.com](https://www.hrbench.com/resource/learn/cost-of-turnover)

[3] hmad, S., & Ahmad, N. (2022). *Employee Attrition and Retention Strategies: A Review of Literature*. ResearchGate. Retrieved from [https://www.researchgate.net](https://www.researchgate.net/publication/362296472)

---

## Business Understanding

### Problem Statement

Tingginya tingkat employee attrition menjadi permasalahan utama bagi perusahaan karena berdampak langsung pada biaya rekrutmen, penurunan produktivitas, dan hilangnya pengetahuan organisasi. Banyak perusahaan kesulitan memahami faktor-faktor yang menyebabkan karyawan meninggalkan perusahaan, seperti kepuasan kerja, kompensasi, atau hubungan kerja. Tanpa analisis berbasis data, langkah yang diambil cenderung reaktif dan tidak efektif. Oleh karena itu, dibutuhkan pendekatan analitik untuk mengidentifikasi penyebab attrition serta memprediksi karyawan yang berisiko tinggi agar strategi retensi dapat dilakukan secara tepat dan preventif.

Berdasarkan hal tersebut, berikut adalah pernyataan masalah yang diangkat:

- **Pernyataan Masalah 1:** Bagaimana mengidentifikasi faktor-faktor penting yang memengaruhi keputusan employee untuk berhenti?
- **Pernyataan Masalah 2:** Bagaimana membangun model prediksi yang mampu memperkirakan kemungkinan seorang employee akan melakukan attrition dengan tingkat akurasi tinggi?  
- **Pernyataan Masalah 3:** Bagaimana menyusun strategi berbasis data untuk menurunkan tingkat attrition?

### Goals

Untuk menjawab pernyataan masalah tersebut, tujuan project ini dirumuskan sebagai berikut:

- **Tujuan 1:** Melakukan eksplorasi dan analisis data karakteristik employee untuk mengidentifikasi pola dan fitur yang berkorelasi tinggi terhadap perilaku attrition.  
- **Tujuan 2:** Membangun model prediktif berbasis machine learning yang mampu menghitung probabilitas attrition dari masing-masing employee.  
- **Tujuan 3:** Memberikan rekomendasi dan rencana aksi yang berbasis pada hasil prediksi model untuk meminimalisir terjadinya attrition.

### Solution Statements

Untuk mencapai tujuan-tujuan tersebut, solusi yang akan diimplementasikan meliputi:

- **Eksperimen Berbagai Algoritma Klasifikasi:**  
  Membangun dan membandingkan performa beberapa algoritma seperti:
  - Decision Tree
  - Random Forest
  - XGBoost
  - Support Vector Machine

- **Optimasi Model dengan Hyperparameter Tuning:**  
  Menggunakan pendekatan seperti GridSearchCV untuk mendapatkan konfigurasi parameter model terbaik. Melalui proses hyperparameter tuning, model dapat mencapai keseimbangan optimal antara akurasi, generalisasi, dan efisiensi prediksi.

- **Evaluasi Model dengan Metrik yang Relevan:**  
  Menggunakan metrik seperti:
  - Accuracy untuk mengukur prediksi keseluruhan  
  - Precision, Recall, F1-Score untuk menilai performa pada kelas churn  
  - ROC-AUC untuk mengevaluasi kemampuan model dalam membedakan kelas  
  - Confusion Matrix untuk melihat distribusi hasil prediksi

- **Analisis Fitur dan Visualisasi:**  
  Menyajikan visualisasi seperti feature importance dan correlation heatmap untuk menginterpretasikan fitur-fitur utama yang berkontribusi terhadap attrition.

### Project Benefits  
Dengan implementasi solusi ini, manfaat utama yang diharapkan antara lain:

- **Meningkatkan Retensi Karyawan** – Model prediksi membantu perusahaan mengenali karyawan berisiko tinggi agar dapat dilakukan intervensi lebih awal.
- **Menekan Biaya Rekrutmen** – Dengan menurunkan tingkat attrition, perusahaan dapat mengurangi biaya perekrutan dan pelatihan karyawan baru.
- **Meningkatkan Produktivitas** – Stabilitas tenaga kerja menjaga kesinambungan kinerja tim dan efisiensi operasional.
- **Mendukung Keputusan Berbasis Data** – Hasil analisis memberikan dasar objektif bagi manajemen dalam membuat kebijakan HR strategis.
- **Meningkatkan Kepuasan Kerja** – Insight dari model dapat digunakan untuk memperbaiki lingkungan kerja dan kesejahteraan karyawan.

---
## Data Undersatnding

### Data Source
Dataset yang digunkan dalam project ini diperoleh dari situs [Kaggle.com](https://www.kaggle.com/competitions/tugas-1-sml-a-2025)
Dataset ini mencakup informasi tentang **1.173 Employee**, yang mencatat berbagai karakteristik employee.

Dataset ini memiliki **35 features** dan **1 target: Attrition**, yang mencakup usia, gaji, jenis pekerjaan, lama waktu di perusahaan, performance rating, dan lainnya. 
Di antara seluruh pelanggan, hanya sekitar **16,15%** yang termasuk dalam kategori attrition (berhenti bekerja). Ketidakseimbangan kelas ini menjadikan proses pelatihan model prediktif sebagai tantangan tersendiri.

### Deskripsi Fitur

| Nama Fitur                         | Deskripsi                                                                 | Tipe Data    |
|------------------------------------|---------------------------------------------------------------------------|--------------|
| `id`                               | ID unik pelanggan                                                         | `object`     |
| `Age`                              | Usia Karyawan (dalam tahun)                                               | `int64`      |
| `BusinessTravel`                   | Frekuensi perjalanan dinas karyawan                                       | `object`     |
| `DailyRate`                        | Gaji harian                                                               | `int64`      |
| `Department`                       | Departemen tempat karywan bekerja                                         | `object`     |
| `DistanceFromHome`                 | Jarak tempat tinggal karyawan ke kantor                                   | `int64`      |
| `Education`                        | Tingkat pendidikan terakhir                                               | `int64`      |
| `EducationField`                   | Bidang studi terakhir                                                     | `object`     |
| `EmployeeCount`                    | Jumlah karyawan                                                           | `int64`      |
| `EmployeeNumber`                   | Nomor unik karyawan dalam sistem HR                                       | `int64`      |
| `EnvironmentSatisfaction`          | Tingkat kepuasan terhadap lingkungan kerja                                | `int64`      |
| `Gender`                           | Jenis kelamin karyawan                                                    | `object`     |
| `HourlyRate`                       | Upah karyawan per jam                                                     | `int64`      |
| `JobInvolvement`                   | Tingkat keterlibatan pekerjaan                                            | `int64`      |
| `JobLevel`                         | Level jabatan karyawan                                                    | `int64`      |
| `JobRole`                          | Posisi / jabatan spesifik karyawan                                        | `object`     |
| `JobSatisfaction`                  | Tingkat kepuasan pekerjaan                                                | `int64`      |
| `MaritalStatus`                    | Status pernikahan karyawan                                                | `object`     |
| `MonthlyIncome`                    | Gaji bulanan karyawan                                                     | `int64`      |
| `MonthlyRate`                      | Tarif bulanan karyawan                                                    | `int64`      |
| `NumCompaniesWorked`               | Jumlah perusahaan tempat karyawan pernah bekerja sebelumnya.              | `int64`      |
| `Over18`                           | Status usia di atas 18 tahun                                              | `boolean`    |
| `OverTime`                         | Apakah karyawan sering lembur                                             | `boolean`    |
| `PercentSalaryHike`                | Persentase kenaikan gaji tahunan terakhir                                 | `int64`      |
| `PerformanceRating`                | Penilaian kinerja terakhir                                                | `int64`      |
| `RelationshipSatisfaction`         | Tingkat kepuasan terhadap hubungan kerja                                  | `int64`      |
| `StandardHours`                    | Jam kerja standar                                                         | `int64`      |
| `StockOptionLevel`                 | Level kepemilikan saham perusahaan                                        | `int64`      |
| `TotalWorkingYears`                | Total tahun pengalaman kerja                                              | `int64`      |
| `TrainingTimesLastYear`            | Jumlah pelatihan yang diikuti dalam setahun terakhir.                     | `int64`      |
| `WorkLifeBalance`                  | Tingkat keseimbangan kerja-hidup                                          | `int64`      |
| `YearsAtCompany`                   | Total tahun bekerja di perusahaan saat ini                                | `int64`      |
| `YearsInCurrentRole`               | Total tahun bekerja di posisi / jabatan saat ini                          | `int64`      |
| `YearsSinceLastPromotion`          | Tahun sejak promosi terakhir                                              | `int64`      |
| `YearsWithCurrManager`             | Tahun bekerja dengan manajer saat ini                                     | `int64`      |
| `Attrition`                        | Apakah karyawan keluar dari perusahaan (1 = Yes/ 0 = No)                  | `int64`      |

### Exploratory Data Analysis

#### Duplicates, Missing Values, and Outliers

Dalam tahap awal pembersihan data, dilakukan pengecekan terhadap **duplikasi data** dan **missing value**. Hasilnya menunjukkan bahwa **tidak terdapat duplikasi data** maupun **missing value** di seluruh kolom fitur maupun target. Hal ini mengindikasikan bahwa dataset sudah lengkap dan tidak memerlukan teknik imputasi lebih lanjut.

| Nama Fitur                         | # Missing    |
|------------------------------------|--------------|
| `id`                               | 0            |
| `Age`                              | 0            |
| `BusinessTravel`                   | 0            |
| `DailyRate`                        | 0            |
| `Department`                       | 0            |
| `DistanceFromHome`                 | 0            |
| `Education`                        | 0            |
| `EducationField`                   | 0            |
| `EmployeeCount`                    | 0            |
| `EmployeeNumber`                   | 0            |
| `EnvironmentSatisfaction`          | 0            |
| `Gender`                           | 0            |
| `HourlyRate`                       | 0            |
| `JobInvolvement`                   | 0            |
| `JobLevel`                         | 0            |
| `JobRole`                          | 0            |
| `JobSatisfaction`                  | 0            |
| `MaritalStatus`                    | 0            |
| `MonthlyIncome`                    | 0            |
| `MonthlyRate`                      | 0            |
| `NumCompaniesWorked`               | 0            |
| `Over18`                           | 0            |
| `OverTime`                         | 0            |
| `PercentSalaryHike`                | 0            |
| `PerformanceRating`                | 0            |
| `RelationshipSatisfaction`         | 0            |
| `StandardHours`                    | 0            |
| `StockOptionLevel`                 | 0            |
| `TotalWorkingYears`                | 0            |
| `TrainingTimesLastYear`            | 0            |
| `WorkLifeBalance`                  | 0            |
| `YearsAtCompany`                   | 0            |
| `YearsInCurrentRole`               | 0            |
| `YearsSinceLastPromotion`          | 0            |
| `YearsWithCurrManager`             | 0            |
| `Attrition`                        | 0            |

Selanjutnya, dilakukan deteksi **outlier** menggunakan metode **Interquartile Range (IQR)** untuk setiap fitur numerik. Hasil analisis menunjukkan bahwa beberapa variabel memiliki jumlah outlier yang cukup signifikan,

| Nama Fitur                         | # Outlier    |
|------------------------------------|--------------|
| `id`                               | 0            |
| `Age`                              | 0            |
| `BusinessTravel`                   | 0            |
| `DailyRate`                        | 0            |
| `Department`                       | 0            |
| `DistanceFromHome`                 | 0            |
| `Education`                        | 0            |
| `EducationField`                   | 0            |
| `EmployeeCount`                    | 0            |
| `EmployeeNumber`                   | 0            |
| `EnvironmentSatisfaction`          | 0            |
| `Gender`                           | 0            |
| `HourlyRate`                       | 0            |
| `JobInvolvement`                   | 0            |
| `JobLevel`                         | 0            |
| `JobRole`                          | 0            |
| `JobSatisfaction`                  | 0            |
| `MaritalStatus`                    | 0            |
| `MonthlyIncome`                    | 86           |
| `MonthlyRate`                      | 0            |
| `NumCompaniesWorked`               | 36           |
| `Over18`                           | 0            |
| `OverTime`                         | 0            |
| `PercentSalaryHike`                | 0            |
| `PerformanceRating`                | 185          |
| `RelationshipSatisfaction`         | 0            |
| `StandardHours`                    | 0            |
| `StockOptionLevel`                 | 66           |
| `TotalWorkingYears`                | 52           |
| `TrainingTimesLastYear`            | 174          |
| `WorkLifeBalance`                  | 0            |
| `YearsAtCompany`                   | 52           |
| `YearsInCurrentRole`               | 16           |
| `YearsSinceLastPromotion`          | 85           |
| `YearsWithCurrManager`             | 10           |
| `Attrition`                        | 0            |

Meskipun demikian, outlier **tidak dihapus** dari dataset. Hal ini dilakukan untuk menjaga **keutuhan informasi**, mengingat data pencilan tersebut mencerminkan kondisi nyata. Menghilangkan outlier justru berisiko menghilangkan pola penting dalam konteks analisis attrition employee.
Sebagai langkah mitigasi terhadap pengaruh outlier, sebelum melakukan pemodelan machine learning, akan dilakukan _scalling_ feature numerik.

---

## Data Preparation

### Label Encoding

Dalam dataset kali ini, label sudah berupa 0 / 1. Sehinggan tidak perlu dilakukan label encoding lagi

| Label | Kategori |
|-------|----------|
| 1     | Yes      |
| 0     | No       |

### Drop Features

Dalam dataset ini, ada beberapa feature yang tidak memiliki variansi seperti 'EmployeeCount', 'StandardHours', 'Over18', dan feature unique feature seperti 'id', dan 'EmployeeNumber'. Kelima features ini tidak akan digunakan dalam analisis kali ini.

### Transformation

Beberapa fitur numerik pada dataset memiliki distribusi yang skewed (tidak simetris), ditandai dengan adanya outlier, yang dapat memengaruhi performa model terutama pada algoritma berbasis jarak seperti SVM.
Untuk menormalkan distribusi tersebut, dilakukan logarithmic transformation (np.log1p) pada fitur-fitur dengan nilai skewness di atas ambang batas (skew_threshold = 1.0).
Transformasi ini membantu mengurangi efek outlier dan membuat data lebih mendekati distribusi normal, sehingga proses pelatihan model menjadi lebih stabil dan akurat.
Dalam analisis kali ini, beberapa features yang dilakukan transformasi logaritmik ialah: 'MonthlyIncome', 'PerformanceRating', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion'

### Scaling and Encoding

Tahap ini dilakukan untuk memastikan seluruh fitur memiliki skala dan format yang sesuai sebelum masuk ke model:

- StandardScaler digunakan untuk menstandarkan fitur numerik (num_cols) agar memiliki mean = 0 dan standard deviation = 1, sehingga model tidak bias terhadap fitur dengan rentang nilai besar.
- OneHotEncoder diterapkan pada fitur kategorikal (cat_cols) untuk mengubah nilai kategori menjadi representasi biner (0/1).
Encoder juga diatur dengan handle_unknown='ignore' agar dapat menangani kategori baru pada data uji tanpa error.
- Hasil dari kedua transformasi ini kemudian digabungkan kembali menjadi satu dataset (X) yang siap digunakan untuk pelatihan model.

### Feature Se;ection

Proses feature selection dilakukan menggunakan Logistic Regression dengan regularisasi L1 (Lasso).
Pendekatan ini membantu menghilangkan fitur yang tidak relevan dengan memberikan bobot nol pada koefisien yang kurang berkontribusi terhadap prediksi.

- Parameter class_weight='balanced' digunakan untuk menangani ketidakseimbangan kelas.
- Hanya fitur dengan bobot penting di atas nilai median (threshold='median') yang dipertahankan.
- Hasilnya adalah subset fitur yang lebih informatif (selected_features), sehingga model menjadi lebih efisien dan berpotensi meningkatkan performa generalisasi.

Hasil akhirnya, didapatkan reduksi feature hingga menjadi **26 features**, yaitu: 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'YearsSinceLastPromotion', 'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'Department_Research & Development', 'Department_Sales', 'EducationField_Human Resources', 'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Female', 'JobRole_Laboratory Technician', 'JobRole_Research Director', 'JobRole_Sales Representative', 'MaritalStatus_Divorced', 'MaritalStatus_Single', 'OverTime_No', 'OverTime_Yes'

### Splitting Dataset

- Menetapkan `stratify = y` sehingga fungsi train_test_split memastikan bahwa proses pemisahan mempertahankan persentase yang sama dari setiap kelas target di set train dan test.

Dataset yang digunakan dalam analisis ini terdiri dari data pelatihan (train) dan data pengujian (test) dengan rpoprosi data test 20% dari total dataset dengan rincian sebagai berikut:

- **Ukuran data fitur (train)**: 940 observasi dengan 26 fitur.
- **Ukuran data target (train)**: 940 observasi.
- **Ukuran data fitur (test)**: 236 observasi dengan 26 fitur.
- **Ukuran data target (test)**: 236 observasi.

### Handling Imbalanced Dataset

Dataset memiliki distribusi kelas yang tidak seimbang, sehingga dilakukan penyeimbangan menggunakan SMOTE-Tomek.

- SMOTE (Synthetic Minority Oversampling Technique) menghasilkan sampel sintetis untuk kelas minoritas agar proporsi kelas lebih seimbang.
- Tomek Links kemudian digunakan untuk menghapus pasangan sampel yang terlalu berdekatan antar kelas, membantu memperjelas batas keputusan (decision boundary).
Hasilnya (X_train_res, y_train_res) merupakan dataset pelatihan yang lebih seimbang dan bersih, sehingga model dapat belajar dengan lebih adil terhadap kedua kelas.

Dataset train menjadi memiliki dimensi:

- **Ukuran data fitur (X_train_res)**: 1574 observasi dengan 26 fitur.
- **Ukuran data target (y_train_res)**: 1574 observasi.

## Modelling with Support Vector Machine

### SVM

Support Vector Machine (SVM) adalah algoritma supervised learning yang digunakan untuk klasifikasi maupun regresi.
Prinsip utamanya adalah mencari garis atau bidang pemisah terbaik (hyperplane) yang memaksimalkan jarak (margin) antara kelas yang berbeda.

SVM bekerja efektif pada data berdimensi tinggi dan tetap kuat terhadap outlier berkat penggunaan support vectors — titik data yang paling berpengaruh dalam menentukan batas keputusan.
Dengan pemilihan kernel yang tepat, SVM juga mampu memetakan data non-linear ke ruang berdimensi lebih tinggi agar dapat dipisahkan dengan baik.

### Modelling

Untuk mendapatkan performa model SVM yang optimal, dilakukan hyperparameter tuning menggunakan GridSearchCV dengan cross-validation sebanyak 5 lipatan.

- Parameter yang diuji adalah C, yaitu koefisien regularisasi yang mengontrol keseimbangan antara margin maximization dan misclassification error.
- Proses pencarian menggunakan metrik ROC AUC sebagai dasar evaluasi agar model seimbang dalam mengenali kedua kelas.
- Hasil pencarian menghasilkan kombinasi parameter terbaik (best_params_) dan nilai ROC AUC tertinggi (best_score_), yang kemudian digunakan sebagai model akhir (best_model).

Hasilnya, GridSerachCV dengan 5 fold menghasilkan:

- Best Parameter: C = 0.1
- Best ROC AUC Score: 0.8771

### Model Calibration and Evaluation

Setelah model terbaik diperoleh dari hasil tuning, dilakukan proses kalibrasi probabilitas menggunakan CalibratedClassifierCV dengan metode isotonic regression.
Langkah ini bertujuan agar nilai probabilitas yang dihasilkan model lebih akurat dan dapat diinterpretasikan dengan lebih baik.

Selanjutnya, dilakukan:

- Pencarian ambang batas optimal (threshold) berdasarkan F1-score untuk menyeimbangkan precision dan recall.
- Evaluasi performa model menggunakan metrik ROC AUC dan classification report, baik untuk model terkalibrasi (val_proba) maupun hasil asli dari SVM (decision_function).
- Hasil evaluasi menunjukkan bahwa kalibrasi membantu meningkatkan kualitas prediksi probabilistik dan stabilitas model dalam membedakan antara kelas positif dan negatif.

Hasilnya:

- Treshhold Optimal (based on F1): 0.712.
- Precision / Recall / F1-score: Kelas 0 → 0.93 / 0.94 / 0.94, Kelas 1 → 0.69 / 0.63 / 0.66
- Accuracy: 0.89
- ROC AUC (calibrated): 0.854

Jika menggunakan threshold default (0.5), performa sedikit berbeda:
- Precision / Recall / F1-score: Kelas 0 → 0.95 / 0.84 / 0.89, Kelas 1 → 0.48 / 0.79 / 0.60
- Accuracy: 0.83
- ROC AUC: 0.851

Hasil ini menunjukkan bahwa penyesuaian threshold meningkatkan keseimbangan F1-score pada kelas minoritas tanpa mengorbankan ROC AUC secara signifikan.









