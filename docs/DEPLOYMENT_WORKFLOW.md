# 🚀 Deployment Workflow: Local Training → VPS API

## Your Setup

- **Local Machine**: Powerful GPUs for training models
- **VPS (Coolify)**: Serving the API to your frontend

---

## 📦 What Goes Where?

### On Local Machine (Your Current Setup)
```
deepfake-text-detector/
├── scripts/               # Training scripts (KEEP LOCAL)
├── data/                  # Training data (KEEP LOCAL)
├── archive/               # Experiments (KEEP LOCAL)
├── models/                # Model code (SYNC TO VPS)
├── utils/                 # Utilities (SYNC TO VPS)
├── api/                   # API code (SYNC TO VPS)
├── saved_models/          # Trained models (SYNC TO VPS)
│   └── *.pkl             # These are the important files!
├── Dockerfile            # Docker config (SYNC TO VPS)
├── requirements.txt      # Dependencies (SYNC TO VPS)
└── start_api.sh          # Startup script (SYNC TO VPS)
```

### On VPS (Minimal Setup)
```
deepfake-backend/         # Only what's needed to serve API
├── api/                  # ← API code
├── models/               # ← Model definitions
├── utils/                # ← Utilities
├── saved_models/         # ← Trained .pkl files (copied from local)
│   └── *.pkl            # The magic happens here!
├── Dockerfile           # ← Docker config
├── requirements.txt     # ← Dependencies
└── start_api.sh         # ← Startup script
```

**You DON'T need on VPS:**
- ❌ Training scripts (`scripts/`)
- ❌ Training data (`data/`)
- ❌ Experiments/archives (`archive/`)
- ❌ Jupyter notebooks

---

## 🔄 Step-by-Step Workflow

### Initial Setup (One Time)

#### 1️⃣ Prepare Local Files

```bash
# On your local machine
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector

# Make sure you have at least one trained model
ls -lh saved_models/*.pkl

# If no models, train one first
python scripts/train_and_save_detector.py
```

#### 2️⃣ Copy to VPS

**Option A: Copy Everything (First Time)**
```bash
# From your local machine
rsync -avz --progress \
  --exclude 'data/' \
  --exclude 'archive/' \
  --exclude 'scripts/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.ipynb' \
  --exclude 'results/' \
  . user@your-vps-ip:/home/user/deepfake-backend/
```

**Option B: Copy Only API Files (Minimal)**
```bash
# Create a deployment package
mkdir -p /tmp/deepfake-deploy
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector

# Copy only what's needed
cp -r api/ /tmp/deepfake-deploy/
cp -r models/ /tmp/deepfake-deploy/
cp -r utils/ /tmp/deepfake-deploy/
cp -r saved_models/ /tmp/deepfake-deploy/
cp Dockerfile /tmp/deepfake-deploy/
cp docker-compose.yml /tmp/deepfake-deploy/
cp requirements.txt /tmp/deepfake-deploy/
cp start_api.sh /tmp/deepfake-deploy/
cp .env.example /tmp/deepfake-deploy/

# Now copy to VPS
cd /tmp
rsync -avz --progress deepfake-deploy/ user@your-vps-ip:/home/user/deepfake-backend/
```

#### 3️⃣ Deploy on Coolify

1. **SSH into your VPS** and verify files:
```bash
ssh user@your-vps-ip
ls -la ~/deepfake-backend/
ls -la ~/deepfake-backend/saved_models/
```

2. **In Coolify Dashboard:**
   - New Resource → Dockerfile
   - Set path: `/home/user/deepfake-backend`
   - Set Dockerfile: `Dockerfile`
   - Add volume: `./saved_models:/app/saved_models:ro`
   - Port: 8000
   - Deploy!

---

### Regular Workflow (After Setup)

#### When You Train a New Model Locally

```bash
# 1. Train model on your local GPU (as usual)
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector
python scripts/train_and_save_detector.py

# 2. Sync ONLY the new model to VPS
rsync -avz --progress \
  saved_models/*.pkl \
  user@your-vps-ip:/home/user/deepfake-backend/saved_models/

# 3. The API will automatically pick up the new model!
# (You may need to restart the container in Coolify or it will load on next request)
```

#### When You Update API Code

```bash
# 1. Test changes locally first
./start_api.sh
./api/test_api.sh

# 2. Sync API code to VPS
rsync -avz --progress \
  api/ \
  user@your-vps-ip:/home/user/deepfake-backend/api/

# 3. Redeploy in Coolify
# (Click "Redeploy" button in Coolify dashboard)
```

---

## 🎯 Quick Commands

### Copy Everything to VPS
```bash
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector

# One command to sync all needed files
rsync -avz --progress \
  --include='api/***' \
  --include='models/***' \
  --include='utils/***' \
  --include='saved_models/***' \
  --include='Dockerfile' \
  --include='docker-compose.yml' \
  --include='requirements.txt' \
  --include='start_api.sh' \
  --include='.env.example' \
  --exclude='*' \
  . user@your-vps-ip:/home/user/deepfake-backend/
```

### Copy Only Models (Quick Update)
```bash
rsync -avz --progress \
  saved_models/ \
  user@your-vps-ip:/home/user/deepfake-backend/saved_models/
```

### Test Before Deploying
```bash
# Test locally first
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector
./start_api.sh

# In another terminal
./api/test_api.sh
```

---

## 💾 Storage Considerations

### Model Files Size
```bash
# Check model sizes
du -sh saved_models/
du -h saved_models/*.pkl
```

**Tips:**
- Each model is typically 50-500MB
- Keep only the best performing models on VPS
- Archive old models locally, don't sync everything

### VPS Storage Management
```bash
# On VPS, check disk usage
ssh user@vps "df -h"
ssh user@vps "du -sh ~/deepfake-backend/saved_models/"

# Remove old models if needed
ssh user@vps "rm ~/deepfake-backend/saved_models/old_model.pkl"
```

---

## 🔐 Git Workflow (Alternative)

If you prefer using Git instead of rsync:

### 1️⃣ Setup (One Time)

```bash
# On local machine
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector

# Create .gitignore for large files
cat > .gitignore << 'EOF'
data/
archive/
results/
*.ipynb
__pycache__/
*.pyc
.env
.DS_Store
EOF

# Initialize git (if not already)
git init
git add api/ models/ utils/ Dockerfile requirements.txt
git commit -m "Initial API setup"

# Add remote (GitHub/GitLab)
git remote add origin https://github.com/yourusername/deepfake-backend.git
git push -u origin main
```

### 2️⃣ In Coolify

- New Resource → Git Repository
- Connect to your repo
- Set branch: `main`
- Coolify will auto-deploy on push!

### 3️⃣ For Models (Too Large for Git)

```bash
# Still use rsync for large model files
rsync -avz saved_models/*.pkl \
  user@vps:/home/user/deepfake-backend/saved_models/
```

Or use **Git LFS** for models:
```bash
git lfs install
git lfs track "*.pkl"
git add saved_models/*.pkl
git commit -m "Add trained models"
git push
```

---

## 🧪 Testing Checklist

### Before Deploying to VPS

- [ ] API runs locally: `./start_api.sh`
- [ ] Tests pass: `./api/test_api.sh`
- [ ] At least one model in `saved_models/`
- [ ] Docker builds: `docker build -t test .`
- [ ] Docker runs: `docker run -p 8000:8000 test`

### After Deploying to VPS

- [ ] Files copied successfully
- [ ] Models present on VPS
- [ ] Coolify deployment successful
- [ ] Health endpoint works: `curl https://your-api.com/health`
- [ ] Model detection works: Test with `/detect`
- [ ] Frontend can reach API

---

## 🆘 Troubleshooting

### "No models found" on VPS

```bash
# Check if models were copied
ssh user@vps "ls -la ~/deepfake-backend/saved_models/"

# Copy them manually if needed
rsync -avz saved_models/*.pkl user@vps:~/deepfake-backend/saved_models/

# Restart container in Coolify
```

### "Import errors" on VPS

```bash
# Make sure all code directories were copied
ssh user@vps "ls -la ~/deepfake-backend/"

# Should see: api/, models/, utils/
# Copy missing directories
```

### API slow on VPS

- VPS may not have GPU (will use CPU - slower)
- First request loads model (takes time)
- Consider using smaller models for VPS
- Check VPS resources in Coolify

---

## 📊 Comparison: Local vs VPS

| Feature | Local (Your GPU Server) | VPS (Coolify) |
|---------|------------------------|---------------|
| **GPU** | ✅ Powerful GPUs | ⚠️ May have GPU or CPU only |
| **Purpose** | Training models | Serving API |
| **Data** | Full training datasets | No data needed |
| **Scripts** | All training scripts | Only API code |
| **Models** | All trained models | Only deployed models |
| **Cost** | Your hardware | Monthly VPS fee |
| **Speed** | Fast training | Fast API responses |

---

## 🎯 Recommended Setup

**Best Practice:**

1. **Keep your current setup for training** (local with powerful GPUs)
2. **Copy only API + trained models to VPS** (for serving)
3. **Use rsync for quick model updates**
4. **Use Git for API code updates** (optional)

This gives you:
- ✅ Fast training locally
- ✅ Public API on VPS
- ✅ Minimal VPS storage usage
- ✅ Easy updates with rsync

---

## 📝 Example Session

```bash
# Morning: Train new model locally
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector
python scripts/train_and_save_detector.py
# New model saved: saved_models/detector_new_model.pkl

# Test locally
./start_api.sh
# Ctrl+C to stop

# Deploy to VPS
rsync -avz saved_models/detector_new_model.pkl \
  user@vps:~/deepfake-backend/saved_models/

# Restart API in Coolify (click Redeploy button)

# Test VPS API
curl https://your-api.com/health
# Should show new model in available_models list

# Done! ✅
```

---

**Summary:** Train locally, deploy API to VPS, sync models as needed! 🚀
