Windows alatt a WSL2 telepítést javaslom. A TensorFlow 2.19 natív Windows alatt nem támogat GPU-t; WSL2-ben viszont CPU-val és NVIDIA GPU-val is használható.

## 1. WSL2 és Ubuntu telepítése

PowerShellben, rendszergazdaként:

```
wsl --install -d Ubuntu
```

Indítsd újra a gépet, majd nyisd meg az Ubuntu alkalmazást, és hozz létre egy Linux-felhasználót.

## 2. Rendszercsomagok telepítése

Az Ubuntu terminálban:

```
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip ffmpeg libgl1 libglib2.0-0
```

## 3. A projekt elhelyezése

Ha a projekt például a Windows C:\felis mappában található:

```
cd /mnt/c/felis
```

A jobb teljesítmény érdekében célszerű inkább a WSL saját fájlrendszerébe másolni vagy klónozni:

```
cd ~
git clone <a-repository-címe> felis
cd felis
```

## 4. Virtuális környezet és az alkalmazás

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[two-stage]'
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --no-deps opencv-python==5.0.0.93
python scripts/check_runtime_dependencies.py --opencv-variant gui
```

Az utolsó parancs kimenet nélkül, nullás visszatérési kóddal jelzi, hogy csak a
GUI-képes OpenCV-változat maradt telepítve. Erre azért van szükség, mert a
PytorchWildlife egyes függőségei a headless csomagnevet kérik, miközben ugyanazt
a `cv2` API-t használják.

Importellenőrzés:

```
python -c "import tensorflow, keras; from PytorchWildlife.models import classification; print('Telepítés rendben')"
```

## 5. Modellfájlok

A szükséges .pt, .keras és .classes.txt fájlokat helyezd a projekt models könyvtárába. Ezek nagy méretük miatt valószínűleg nincsenek benne a Git repositoryban.

Például:

```
felis/
├── models/
│   ├── best_27.pt
│   ├── best_28.pt
│   ├── 4_camtrap.keras
│   ├── 4_camtrap.classes.txt
│   ├── 2_artiodactyla.keras
│   ├── 2_artiodactyla.classes.txt
│   ├── 2_carnivora.keras
│   └── 2_carnivora.classes.txt
```

## 6. Windows könyvtárak megadása

A YAML-konfigurációban Linux/WSL útvonalakat kell használni. Például a Windows alatti

```
C:\kameracsapda\raw
```

WSL-ben:

```
/mnt/c/kameracsapda/raw
```

Példakonfiguráció:

```
input_root: /mnt/c/kameracsapda/raw
output_root: /mnt/c/kameracsapda/results
username: gabor
camera_id: bzs4b
footage_date: "20180307"

strategy: two_stage
detector: best_27
classifier: deepfaune_classifier

model_path: /home/felhasznalo/felis/models/best_27.pt
models_dir: /home/felhasznalo/felis/models

device: cpu
imgsz: 1280
conf: 0.25
iou: 0.45
save_frames: true
```

Futtatás:

```
source .venv/bin/activate
felis run --config=.felis.local.yml
```

NVIDIA GPU esetén a Windows NVIDIA-driver telepítése után WSL-ben ezt állítsd:

```
device: cuda:0
```

Ellenőrzés:

```
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

A hivatalos TensorFlow dokumentáció szerint natív Windowson a TensorFlow 2.10 volt az utolsó GPU-t támogató kiadás; újabb TensorFlow-verzióhoz a WSL2 az ajánlott környezet. (TensorFlow
telepítési útmutató)
