# 🎥 Lip Reading Module - Proiect SOL10

## 📝 Descriere
Acest modul realizeaza recunoastere vizuala a vorbirii (*lip reading*).  
Codul preia la intrare un fisier video **fara audio** (decupat in jurul gurii vorbitorului si aliniat cu ajutorul modulului `lip_reading_video_preprocessing`) si prezice transcrierea.  

---

## ⚙️ Cerinte

- 🐍 **Python 3.10**  
- 🔥 **PyTorch 2.1.2 (CUDA 12.1)**  
- 💻 **NVIDIA GPU** + driver compatibil cu CUDA 12.1  

Pentru reproducere sigura se recomanda folosirea urmatoarelor instructiuni pentru instalarea librariilor:

```bash
# 📦 Instaleaza pachete de sistem
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg

# 📦 Instaleaza dependente Python
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2"
pip install "transformers==4.51.1" "accelerate==1.6.0"
pip install "omegaconf==2.0.6"
pip install "opencv-python==4.11.0.86"
pip install "hydra-core==1.0.7"
pip install "python-speech-features==0.6"
pip install "scipy==1.15.2"
pip install "peft==0.15.1"
pip install "bitsandbytes==0.45.5"
pip install "imageio==2.37.0"
pip install "imageio-ffmpeg==0.6.0"
pip install --upgrade --force-reinstall "numpy==1.23.5"
```

---
## ▶️ Utilizare

Rulare script principal pentru recunoașterea vorbirii dintr-un videoclip fără audio:

```bash
python main_lip_reading.py /cale/catre/video_NoAudio.mp4
```

### 📤 Output

Exemplu de rezultat afișat în consolă:

```text
RUN lip_reading_demo_module - V03 14.11.2026

Video: ./samples/Video_1_NoAudio.mp4
Predictie   : <transcrierea prezisa>
```

### 📊 Evaluare pe setul de validare

Pentru evaluarea automată a tuturor videoclipurilor:

```bash
python main_lip_reading_test_DB.py --db_path /cale/catre/VALIDATION/
```

Rezultatele sunt salvate automat în directorul `results/`, într-un fișier JSON al cărui nume conține data și ora:

```text
results/Rezultate_LipReading_Model_14_11_2026_23_53_00.json
```

## 🧠 Modele folosite

Modelele folosite in cadrul proiectului pot fi descarcate de la acest link:  
[📥 Download modele](http://tiny.cc/Sol10)

---

## ⚠️ Limitari

- La intrare sunt acceptate doar video-uri **mai mici de 15 secunde**, preprocesate cu ajutorul modulului `lip_reading_video_preprocessing`.  

---

## 🕑 Versiuni semnificative

- **V01_01 – 19.08.2025**: prima versiune functionala
- **V01_02 – 01.11.2025**: introducerea scriptului `main_lip_reading_test_DB.py` care permite testarea pe toate videoclipurile din setul de validare si calculeaza automat scorurile de performanta. 
- **V03 – 14.11.2026**: distributie de deploy.

---

## 👨‍💻 Autor
- [bcmocanu](https://github.com/bcmocanu)
