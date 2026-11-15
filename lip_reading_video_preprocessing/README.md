# 🎥 Preprocesare Video pentru Lip Reading - Proiect SOL10

## 📝 Descriere
Acest modul din cadrul proiectului SOL10 realizeaza preprocesarea fisierelor video pentru task-ul de *lip reading* (citirea pe buze) - modulul `lip_reading_demo_module`.  
Scopul este de a transforma un video brut intr-un set de segmente video aliniate pe fete, decupate pe zona gurii, ce poate fi folosit pentru testarea modulului `lip_reading_demo_module`.

---

## 🛠️ Functionalitati

- Conversie video la **25 fps** cu `ffmpeg`
- Detectarea fetelor, landmark-urilor si pozitiei capului
- Alinierea fetelor pe baza landmark-uri
- Decupare fete la dimensiune fixa
- Salvarea segmentelor video cu zona gurii (mp4)

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

Rulare script principal:

```bash
python lip_reading_video_preprocessing.py /cale/catre/video.mp4
```

---

### 📤 Output

- Creeaza un director nou cu acelasi nume ca video-ul de intrare (in aceeasi locatie)
- Salveaza segmentele video preprocesate (ex. `video_ID0_SEG0_NoAudio.mp4`), cu fetele detectate, decupate si aliniate.

---

## 🧠 Modele folosite

Modelele folosite in cadrul proiectului pot fi descarcate de la acest link:  
[📥 Download modele](http://tiny.cc/Sol10)

---

## ⚠️ Limitari

- La intrare sunt acceptate doar video-uri **mai mici de 15 secunde**. Aceasta este o constraingere a modului `lip_reading_demo_module`.

---

## 🕑 Versiuni semnificative

- **V01_01 – 18.08.2025**: prima versiune functionala

---

## 👨‍💻 Autor
- [bcmocanu](https://github.com/bcmocanu)
