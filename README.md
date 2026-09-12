# Setup Project Neural Network

Panduan ini menjelaskan langkah-langkah yang perlu dilakukan setelah melakukan clone project **belajar-neural-network-python**.

## 1. Clone Repository

Clone repository ke komputer:

```bash
git clone <URL_REPOSITORY>
```

Masuk ke folder project:

```bash
cd belajar-neural-network-python
```

> Ganti `<URL_REPOSITORY>` dengan URL repository Git yang digunakan.

## 2. Pastikan Python Sudah Terinstall

Cek versi Python:

```bash
python --version
```

atau pada beberapa sistem:

```bash
py --version
```

Project ini menggunakan Python. Pastikan versi Python yang digunakan sesuai dengan versi yang ditetapkan oleh project.

## 3. Buat Virtual Environment

Disarankan membuat virtual environment agar dependency project terisolasi.

Windows:

```bash
python -m venv .venv
```

Aktifkan:

```powershell
.venv\Scripts\Activate.ps1
```

Jika menggunakan Command Prompt:

```cmd
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Jika berhasil, terminal biasanya akan menampilkan:

```text
(.venv)
```

di awal prompt.

## 4. Install Dependency

Setelah virtual environment aktif, install seluruh dependency dari `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Jika ingin memperbarui pip terlebih dahulu:

```bash
python -m pip install --upgrade pip
```

Kemudian:

```bash
python -m pip install -r requirements.txt
```

## 5. Periksa Struktur Project

Setelah clone, struktur utama project seharusnya kurang lebih seperti:

```text
belajar-neural-network-python/
├── examples/
│   └── main.py
├── neural_network/
│   ├── core/
│   ├── data/
│   ├── evaluating/
│   ├── training/
│   ├── utils/
│   └── visualization/
│       ├── __init__.py
│       └── plot.py
├── plots/
│   └── .gitkeep
├── tests/
├── .gitignore
└── requirements.txt
```

Folder `plots/` digunakan sebagai tempat menyimpan gambar hasil visualisasi training/evaluating.

File `.gitkeep` digunakan agar folder `plots/` tetap tersimpan di Git walaupun belum memiliki file gambar.

## 6. Jalankan Contoh Program

Program contoh berada di:

```text
examples/main.py
```

Dari root project, jalankan:

```bash
python -m examples.main
```

Jika project menggunakan import package `neural_network`, menjalankan file dari root project membantu Python menemukan package tersebut dengan benar.
