# Brain MRI Tumor Detection Project

Yapay zeka destekli beyin MR görüntüsü analizi projesi. Tümör tespiti, boyutu ve tedavi önerileri sunar.

## 🚀 Özellikler

- **ResNet18** tabanlı derin öğrenme modeli
- **Grad-CAM** heatmap görselleştirmesi
- Modern ve responsive **web arayüzü**
- **Drag & drop** dosya yükleme
- **Tümör sınıflandırma** (Glioma/Meningioma/Tumor)
- **Boyut ve evre tahmini**
- **Tedavi önerileri** (demo amaçlı)
- **Docker** desteği

## 📊 Dataset

Veri seti konumu:
```
data/Brain_Cancer raw MRI data/Brain_Cancer/
```

Sınıflar:
- `brain_glioma` - Glioma
- `brain_menin` - Meningioma  
- `brain_tumor` - Genel tümör

> **Not:** Bu veri seti "No_Tumor" sınıfı içermediği için API şu an `tumor_present=true` varsayar.

## 🛠 Kurulum

### Geliştirme Ortamı

```bash
# Python sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Docker ile

```bash
# Build ve çalıştır
docker-compose up --build

# Veya sadece build
docker-compose build
```

## 🎯 Model Eğitimi

```bash
python training/train_classifier.py
```

Model kaydedilir: `models/brain_mri_classifier.pth`

## 🌐 API Kullanımı

### Sunucuyu Başlat

```bash
# Geliştirme modu
uvicorn backend.main:app --reload --port 8000

# Production modu
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /` - Web arayüzü
- `POST /predict` - MRI analizi
- `GET /heatmaps/{filename}` - Heatmap görüntüsü

### Predict Örneği

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/mri_image.jpg"
```

## 🎨 Web Arayüzü

Modern web arayüzü özellikleri:
- **Responsive tasarım** (mobil uyumlu)
- **Drag & drop** yükleme
- **Anlık analiz sonuçları**
- **Görsel heatmap** gösterimi
- **Detaylı raporlama**

## 📱 Demo

Uygulama çalıştırıldığında http://localhost:8000 adresinden erişilebilir.

## ⚠️ Yasal Uyarı

Bu uygulama **eğitim ve demo amaçlıdır**. Çıktılar **tıbbi tanı veya tedavi tavsiyesi yerine geçmez**. Gerçek tıbbi uygulamalar için mutlaka uzman hekime danışın.

## 🔧 Teknolojiler

- **Backend:** FastAPI, PyTorch, OpenCV
- **Frontend:** HTML5, TailwindCSS, JavaScript
- **Model:** ResNet18, Grad-CAM
- **Deployment:** Docker, Docker Compose

## 📁 Proje Yapısı

```
brain_mri_project/
├── backend/
│   ├── main.py              # FastAPI uygulaması
│   ├── inference.py         # Model tahmin motoru
│   └── templates/
│       └── index.html       # Web arayüzü
├── models/
│   ├── brain_mri_classifier.pth  # Eğitilmiş model
│   └── heatmaps/            # Grad-CAM çıktıları
├── training/                # Eğitim scriptleri
├── data/                    # Veri seti
├── requirements.txt         # Python bağımlılıkları
├── Dockerfile              # Docker imajı
├── docker-compose.yml      # Docker Compose
└── README.md               # Bu dosya
```

## 🤝 Katkı

Katkıda bulunmak için:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull request açın

## 📄 Lisans

Bu proje eğitim amaçlıdır. Tıbbi kullanım için uygun değildir.

