# NICHD_Classifier

_Automated question-type classification for forensic interviews and trial testimony in child sexual abuse cases._  
This repository is a **fork** of **USC-Lyon-Lab/question_classification** and adds a simple CLI (`predictor_cli.py`) to run inference from the command line.

---

## 🚀 Quick start

### 0) Clone the repo
```bash
git clone https://github.com/Cyber-Vadok/NICHD_Classifier.git
cd NICHD_Classifier
```

### 1) Install **Git LFS** (required to download the model weights)
GitHub blocks files >100 MB unless they’re served with **Git Large File Storage (LFS)**.  
Install LFS following the official guide, then enable it for your user:

```bash
# Install: follow the platform-specific instructions in the site below
# https://git-lfs.com/

git lfs install          # run once per user
git lfs fetch --all      # ensure LFS objects are pulled
git lfs pull
```
> If you skip this step, the model files in `roberta_master2_hf/` will be missing or replaced by small pointer files.

### 2) Create a Python virtual environment (`.venv`)
**Linux / macOS (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the classifier (CLI)
Basic usage:
```bash
python predictor_cli.py --help
python predictor_cli.py
```
---

## 📁 Repository layout (key files)

- `predictor_cli.py` — Command-line entry point for inference  
- `roberta_master2_hf/` — Model directory tracked via Git LFS  
- `requirements.txt` — Python dependencies  

---

## 🔍 Labels

The model classifies questions into four categories consistent with the upstream description (e.g., invitations, WH-questions, option-posing, non-questions). See the original repository for background and context.

---

## 📜 Citation & Credits

- **Upstream project:** USC-Lyon-Lab, _question_classification_. This repository make the model works and adds a CLI.  
- **This fork:** Cyber-Vadok/NICHD_Classifier.

If you use this code in academic work, please also cite the related research from the upstream and any associated publications.

---

## 📝 License

See the upstream repository licensing details.

---
