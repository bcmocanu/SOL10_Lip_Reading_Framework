# 🎥 Lip reading module

## 📝 Descriere
Aplicație locală pentru citirea de pe buze. Necesită Ubuntu x86_64,
un driver NVIDIA funcțional, mediul Python și internet la
prima instalare.

## Prima instalare

Din directorul proiectului, ca utilizator obișnuit, fără `sudo`:

```bash
./setupFramework
./startFramework
```

Setup verifică modelele, driverul, bibliotecile native și mediul Python. Creează
`LipsWebEnv/` cu Python 3.10 și instalează numai acolo dependențele fixate, inclusiv
PyTorch 2.1.2 cu bibliotecile CUDA 12.1. Aceste biblioteci NU sunt driverul NVIDIA.
Nu instalează CUDA Toolkit în sistem, nu modifică driverul și nu configurează Docker.

Dacă lipsește un Python 3.10 utilizabil, propune descărcarea unui Python propriu în
`.runtime/python/`, fără înlocuirea Python-ului Ubuntu. Similar, Node.js poate fi
descărcat în `.runtime/node/` numai pentru compilarea interfeței. Descărcările
acestor două componente sunt verificate prin SHA-256 și cer confirmare.

FFmpeg și bibliotecile de sistem existente sunt reutilizate. Pentru componente
lipsă, setup afișează planul APT și cere acord înainte de `sudo`; refuză eliminarea
sau actualizarea pachetelor existente. Nu adaugă depozite APT și nu repornește servicii.
O problemă a driverului sau un plan APT incompatibil necesită verificarea administratorului.

La final, setup testează importurile, operații CUDA/NF4 și citirea/scrierea H.264.
Ponderile complete ale recunoașterii se încarcă la prima transcriere. Procesarea unui
clip real este verificarea finală a instalării. Setup poate fi reluat după o întrerupere;
oprește mai întâi această aplicație dacă este pornită.

## Utilizare

```bash
./startFramework
./stopFramework
```

Start afișează adresa, de regulă `http://127.0.0.1:5173`. Dacă portul este ocupat,
alege altul fără să oprească serviciul existent. O a doua pornire nu creează o
instanță duplicată. Stop păstrează modelele și datele. Niciuna dintre aceste două
comenzi nu instalează pachete sau cere `sudo`.

Configurația locală este `framework/config.json`: `gpu` este indexul NVIDIA fizic,
`port` este portul preferat, iar `bind` este adresa de ascultare. Implicit, accesul
este limitat la calculatorul local. Pentru acces din rețea, `bind` poate fi schimbat
în `0.0.0.0`; aplicația nu include autentificare sau HTTPS și nu trebuie expusă direct
pe internet. După modificarea configurației, oprește și pornește aplicația.

Modelele sunt citite local; după instalare nu este necesar internet pentru inferență.
GPU-ul și memoria sa rămân resurse comune: alte aplicații pot reduce memoria disponibilă.
Jurnalul aplicației este `.framework/logs/application.log`.

## Ștergere

```bash
./deleteFramework
```

**Comanda cere confirmare și șterge întreg directorul acestei instalări, inclusiv
MODELELE, încărcările, rezultatele, mediul Python și cache-urile locale.** Oprește
mai întâi numai procesele identificate ca proprii.

## 🕑 Versiuni semnificative

- **V01_01 – 20.09.2026**: prima versiune functionala
- **V01_02 – 15.11.2026**: distributie de deploy.

---

## 👨‍💻 Autor
- [bcmocanu](https://github.com/bcmocanu)