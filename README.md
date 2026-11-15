# 🎥 SOL10_Lip_Reading_Framework

## 📝 Descriere

SOL10_Lip_Reading_Framework realizeaza **recunoasterea vizuala a vorbirii in limba romana** din videoclipuri fara audio. Fluxul principal este: **video brut → preprocesare (zona gurii) → model de lip reading → transcriere text**. Proiectul include si o interfata web, evaluare pe o baza de date si antrenare in doua etape.

---

## 📦 Modulele proiectului

**Rolul fiecarui modul:**

- **`lip_reading_video_preprocessing`** – converteste video-ul la 25 FPS, detecteaza si aliniaza fata, decupeaza zona gurii la 96×96 si exporta clipuri `*_NoAudio.mp4`.
- **`lip_reading_demo_module`** – prezice textul dintr-un clip deja preprocesat; permite si evaluarea automata pe setul de validare.
- **`lip_reading_demo_web`** – aplicatie locala cu interfata web: incarcare video, alegerea segmentului si a fetei, preprocesare si transcriere. Include deja codul modulului de inferenta in `core/lip_reading_demo_module/`.
- **`lip_reading_training_module`** – Realizeaza antrenarea modelelor necesare pentru citirea de pe buze.

---

## 🧠 Descarcarea si copierea modelelor

Modelele se descarca de aici: **[📥 Download modele SOL10](http://tiny.cc/Sol10)**. Copiaza **continutul** fiecarui folder-sursa in destinatiile de mai jos, pastrand subdirectoarele si numele fisierelor.

| Din `ModeleFinale/` | In directoarele proiectului |
|---|---|
| `ModelsForPreprocessing/` | `lip_reading_video_preprocessing/Models/`<br>`lip_reading_demo_web/core/models/` |
| `ModelsForPredictions/` | `lip_reading_demo_module/Models/`<br>`lip_reading_demo_web/core/lip_reading_demo_module/Models/` |
| `ModelsForStartTraining/` | `lip_reading_training_module/Models/` |

**Important:** In aplicatia web, codul de inferenta este deja inclus; **nu** trebuie copiat din nou intregul `lip_reading_demo_module`, ci doar modelele in directorul indicat.

---

## ⚙️ Cerinte

- **Linux / Ubuntu, Python 3.10, GPU NVIDIA**, driver compatibil cu **CUDA 12.1**, **PyTorch 2.1.2** si **FFmpeg**.
- Pentru modulele Python independente, instaleaza dependentele conform README-urilor lor; modulul de antrenare foloseste si `requirements.txt`.
- Aplicatia web isi pregateste propriul mediu si dependentele prin `./setupFramework` (internet necesar la prima instalare).

---

## ▶️ Rulare

Comenzile se executa **din directorul modulului respectiv**, dupa pregatirea mediului si copierea modelelor.

**1. Preprocesare video** – transforma un video brut in clipuri fara audio, pregatite pentru predictie:

```bash
cd lip_reading_video_preprocessing
python main_lips_video_preprocessing_noAudio.py /cale/catre/video.mp4
```

Rezultatele se salveaza intr-un subdirector cu numele videoclipului, de exemplu `video/video_ID0_SEG0_NoAudio.mp4`.

**2. Predictie si evaluare (CLI)** – foloseste un clip rezultat din preprocesare:

```bash
cd lip_reading_demo_module
python main_lip_reading.py /cale/catre/video_ID0_SEG0_NoAudio.mp4

# Optional: evaluare pe setul de validare
python main_lip_reading_test_DB.py --db_path /cale/catre/VALIDATION/
```

Evaluarea salveaza rezultatele JSON in `results/`.

**3. Interfata web** – include preprocesarea si predictia intr-o singura aplicatie:

```bash
cd lip_reading_demo_web
./setupFramework    # doar la prima instalare
./startFramework
# Deschide adresa afisata (de regula http://127.0.0.1:5173)
./stopFramework     # oprirea aplicatiei
```

**4. Antrenare** – configureaza mai intai `paths.db_root` in `src/configs/stage1.yaml` si `stage2.yaml`, apoi:

```bash
cd lip_reading_training_module
python setupFramework.py
python train.py --stage 1
python train.py --stage 2
```

Stage 2 necesita `Models_Trained/romanizer/checkpoint_best.pt` produs de Stage 1. Rezultatele sunt salvate separat in `Models_Trained/romanizer/` si `Models_Trained/avsr/`.

---

## ⚠️ Date si limitari

- Pentru inferenta, clipurile trebuie sa fie **fara audio, 25 FPS, 96×96** si sa aiba **sub 15 secunde**.
- Pentru antrenare, baza de date contine video-uri deja preprocesate, fisiere audio mono **16 kHz** (Stage 1) si fisiere JSON cu transcrierea. Structura exacta este documentata in `lip_reading_training_module/README.md`.
- Fiecare modul contine propriul `README.md` cu detalii despre instalare, configurare si utilizare.

---

## 🕑 Actualizare

Ultima actualizare - 15.11.2026

---

## 👨‍💻 Autor

- [bcmocanu](https://github.com/bcmocanu)
