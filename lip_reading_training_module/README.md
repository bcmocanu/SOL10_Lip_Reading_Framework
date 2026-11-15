# Lip Reading Training Module

Stage 1 antreneaza AV-Romanizer din video + audio.

Stage 2 antreneaza adaptorul si LoRA pentru Llama folosind numai video.

Din directorul proiectului rulati pe rand:

```bash
python setupFramework.py
python train.py --stage 1
python train.py --stage 2
```

Comenzile se ruleaza succesiv. Stage 2 se porneste dupa ce stage 1 a produs `checkpoint_best.pt`.

Pentru un experiment nou, folderul Models_Trained nu trebuie sa contina alte modele *.pt.

## Modele si rezultate

```text
Models/
  avhubert/large_vox_iter5.pt
  romanizer/checkpoint_best.pt
  avsr/checkpoint_best.pt
  llama/ 

Models_Trained/
  romanizer/                     # Rezultatele stage 1
    checkpoint_best.pt
    checkpoint_last.pt
    train.log
  avsr/                          # Rezultatele stage 2
    checkpoint_best.pt
    checkpoint_last.pt
    train.log
```

`Models` contine numai intrari si nu este folosit pentru salvarea rezultatelor.

Fisierele `src/configs/stage1.yaml` si `src/configs/stage2.yaml` contin parametrii de configurare.

`paths.output_dir` stabileste directorul rezultatelor. `checkpoint_last.pt` este pentru reluare;

Stage 2 foloseste implicit `Models_Trained/romanizer/checkpoint_best.pt`, fara fallback la romanizerul original. Nu inlocuiti acest fisier pe parcursul unei antrenari stage 2. Identitatea lui este verificata si salvata in checkpoint.

## Configuratii si date

Caile relative din YAML sunt raportate la radacina proiectului. Configuratiile au implicit:

```yaml
paths:
  db_root: /media/user/S4A/SOL10/DB_V2    #trebuie schimbata calea!!!
  models_dir: Models
  manifest_dir: data/manifests
```

Cele doua etape folosesc aceleasi `db_root`, `manifest_dir` si optiuni `data`. Limba este fixata la `ron` / `Romanian`. In stage 2, `zero_shot_lang="no"` si `back_trans_prob=0` nu reprezinta codul limbii datelor.

Structura unui exemplu:

```text
DB_V2/TRAIN/006/00050138/
  00050138_NoAudio.mp4
  00050138_Audio.wav
  00050138_text.json
```

Videoclipurile trebuie sa contina deja regiunea gurii, la 25 FPS, cu dimensiuni de 96x96. Nu se efectueaza detectie faciala, aliniere sau conversie FPS. Audio trebuie sa fie mono, 16 kHz. JSON-ul contine `{"text": "transcrierea"}`, pe un singur rand. Stage 2 nu citeste audio.

### Pregatirea proiectului

`setupFramework.py` verifica mediul, bibliotecile, configuratiile, fisierele modelelor si tokenizerul local. Extensia originala Cython se compileaza doar daca lipseste, nu poate fi importata sau sursa este mai noua. Fisierele intermediare sunt in `src/build`, iar extensia `.so` este in `fairseq/data`.

Pentru ambele etape se cauta JSON-urile, se verifica metadatele video, primul cadru si metadatele audio, apoi se pregatesc manifestele comune. Progresul este afisat inclusiv in timpul cautarii directoarelor. Verificarea primului cadru nu garanteaza integritatea tuturor cadrelor.

Se produc numai `train.tsv/.wrd/.ltr` si `validation.tsv/.wrd/.ltr`. Cand datele si manifestele sunt coerente, se pastreaza fisierele si ordinea lor. Formatul are patru coloane: limba, cale video, cale WAV real, numar de cadre video, dupa linia radacina `/`.

La modificarea bazei rulati din nou setup-ul. Daca exista `checkpoint_last.pt`, datele unei antrenari nu sunt rescrise. Pentru alta baza folositi un experiment cu directoare de rezultate goale.

Setup-ul nu incarca ponderile complete si nu executa forward/backward. Lipsa romanizerului antrenat inainte de stage 1 este anuntata si nu blocheaza pregatirea; fisierul devine obligatoriu la pornirea stage 2. Comenzile de training verifica manifestele fara sa rescaneze videoclipurile.

## Learning rate si durata

In stage 1, `optimization.max_epoch` stabileste bugetul principal (implicit 50). Inainte de prima actualizare, frameworkul construieste samplerul original din metadate, dupa filtrare, si numara minibatchurile. Acest calcul nu decodeaza video/audio, nu avanseaza iteratorul de antrenare si pastreaza starea generatoarelor aleatoare.

Numarul planificat de actualizari este suma, pe epoci, a numarului de grupuri de minibatchuri determinate de `update_freq`. Este inclus si ultimul grup incomplet. Nu se presupune un numar fix de exemple pe minibatch.

Pentru stage 1:

| Faza | Proportie din actualizari | Learning rate |
|---|---:|---|
| Warmup liniar | 5% | 0.000001 -> 0.0001 |
| Mentinere | 15% | 0.0001 |
| Scadere exponentiala | 80% | 0.0001 -> 0.000001 |

Fazele sunt calculate din bugetul efectiv, cu rotunjirea atribuita fazei de scadere. Pentru 50 de epoci si acumulare constanta, limitele sunt aproximativ epocile 2.5, 10 si 50. Numarul exact de actualizari si limitele fazelor apar in log. Proportiile se editeaza in `lr_scheduler.phase_ratio`; nu sunt valori optime garantate pentru orice baza.

Stage 2 pastreaza schedulerul cosine, warmup de 500 de actualizari, LR maxim 0.00005 si `min_lr=0.00001`. Limita suplimentara `max_update=130000` este pastrata. Bugetul efectiv este minimul dintre bugetul epocilor si aceasta limita; perioada cosine acopera actualizarile ramase dupa warmup. Cand bugetul este prea mic pentru warmup, configuratia este respinsa cu un mesaj explicit.

Schedulerul avanseaza numai la actualizari reusite. Overflow sau minibatchuri omise pot face ca numarul executat pana la limita de epoci sa fie putin mai mic decat cel planificat. Nu se modifica automat durata fazelor in timpul rularii.

## Fine-tuning si reluare

Fara `checkpoint_last.pt` in directorul rezultatelor, `paths.initial_checkpoint` furnizeaza ponderile pentru un fine-tuning nou. Optimizerul, schedulerul, contoarele si iteratorul pornesc de la inceput. Implicit se folosesc modelele de start din `Models/romanizer` si `Models/avsr`.

Cu `checkpoint_last.pt` prezent, se restaureaza starea antrenarii si planul de learning rate salvat. Planul nu se recalculeaza din epoca la care s-a oprit procesul. Schimbarea bugetului, a setarilor relevante de batching sau a manifestelor TRAIN este respinsa in loc sa mute tacut fazele schedulerului. Planul si identificatorii manifestelor sunt stocati in checkpoint, fara fisiere auxiliare.

Un checkpoint de reluare trebuie sa contina acest plan. Checkpointurile preantrenate sau provenite din alte implementari pot fi folosite ca `paths.initial_checkpoint` pentru un fine-tuning nou, nu ca `checkpoint_last.pt` pentru a schimba implicit schema unei rulari existente.

Pentru un romanizer nou doar din AV-HuBERT, setati `paths.initial_checkpoint: null` in stage1.yaml si folositi un director de rezultate gol. Encoderul porneste din `Models/avhubert/large_vox_iter5.pt`; proiectia CTC este initializata de la zero. Alternativa este comentata in YAML. Nu se initializeaza aleator intregul AV-HuBERT.

La un fine-tuning nou stage 2, romanizerul ales este pastrat integral, inclusiv statisticile BatchNorm, fara a le inlocui cu cele din checkpointul AVSR de start. La reluare, se restaureaza starea salvata de stage 2 si se verifica identitatea romanizerului prin SHA-256. Nu antrenati stage 1 in acelasi director in timp ce stage 2 il foloseste.

Checkpointurile stage 2 necesita romanizerul selectat si Llama/tokenizerul de baza.
## Mediu

Linux, Python 3.10, PyTorch 2.1.2 + CUDA 12.1 si versiunile din `requirements.txt`. Nu instalati Fairseq din alt repository: subsetul necesar este inclus local. Compilarea necesita un compilator C++ si headerele Python.

Doar pentru crearea unui mediu nou, cu acces la pachete:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==24.0' setuptools wheel
python -m pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Pe Ubuntu se pot folosi `build-essential` si pachetul de dezvoltare corespunzator Python 3.10. Pentru instalare fara internet, pachetele trebuie pregatite anterior sau se foloseste mediul existent. Setup-ul nu instaleaza automat biblioteci. Dupa instalarea mediului si copierea modelelor, pregatirea si antrenarea functioneaza offline.