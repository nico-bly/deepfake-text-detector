# 🎯 TLDR - Quick Answer to Your Question

## Your Question:
> "Should I put all this code on my VPS? It's currently on my local GPUs where I train models."

## Answer: **NO! Keep training local, deploy API to VPS** ✅

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL MACHINE (Your Current Setup - Keep This!)            │
│  ═══════════════════════════════════════════════════════    │
│                                                              │
│  🔥 Powerful GPUs for Training                              │
│                                                              │
│  deepfake-text-detector/                                     │
│  ├── scripts/           ← Training scripts (STAY LOCAL)     │
│  ├── data/              ← Training data (STAY LOCAL)        │
│  ├── archive/           ← Experiments (STAY LOCAL)          │
│  ├── models/            ← Model code (COPY TO VPS)          │
│  ├── utils/             ← Utilities (COPY TO VPS)           │
│  ├── api/               ← API code (COPY TO VPS)            │
│  └── saved_models/      ← Trained .pkl files (COPY TO VPS)  │
│      └── *.pkl          ← These are what matters!           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ rsync / scp
                            │ (Copy API + Models)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  VPS (Coolify - Deploy Here!)                               │
│  ═══════════════════════════════════                         │
│                                                              │
│  🌐 Public API Server                                       │
│                                                              │
│  deepfake-backend/                                           │
│  ├── api/               ← API code (FROM LOCAL)             │
│  ├── models/            ← Model definitions (FROM LOCAL)    │
│  ├── utils/             ← Utilities (FROM LOCAL)            │
│  ├── saved_models/      ← Trained models (FROM LOCAL)       │
│  │   └── *.pkl          ← The magic! 🎯                     │
│  ├── Dockerfile         ← Docker config                     │
│  └── requirements.txt   ← Dependencies                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND (Coolify - Also on VPS)                           │
│  ════════════════════════════════════                        │
│                                                              │
│  🎨 React/Vite App                                          │
│  Calls: https://api.yourdomain.com/detect                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 What to Do

### ✅ On Your LOCAL Machine (Now)

```bash
# You're here: /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector

# 1. Keep training models (as you normally do)
python scripts/train_and_save_detector.py

# 2. Test API locally (optional)
./start_api.sh

# 3. Deploy to VPS when ready
./deploy_to_vps.sh user@your-vps-ip:/home/user/deepfake-backend
```

### ✅ On Your VPS (Coolify)

```bash
# After copying files, just deploy in Coolify:
# 1. New Resource → Dockerfile
# 2. Point to: /home/user/deepfake-backend
# 3. Add volume: ./saved_models:/app/saved_models
# 4. Deploy!
```

---

## 🎯 Three Simple Commands

```bash
# 1. Test locally
./start_api.sh

# 2. Copy to VPS
./deploy_to_vps.sh user@vps:/path/to/backend

# 3. Deploy in Coolify (use dashboard)
```

---

## 💡 Key Points

1. **Training stays LOCAL** (you have the GPUs) 🏋️
2. **API goes to VPS** (public access) 🌐
3. **Models sync from local to VPS** (rsync) 📦
4. **No training code on VPS** (not needed) ✂️

---

## 📦 What Gets Copied to VPS?

### ✅ COPY (Small Files)
- `api/` - API code (~50KB)
- `models/` - Model definitions (~500KB)
- `utils/` - Utilities (~100KB)
- `Dockerfile` - Docker config
- `requirements.txt` - Dependencies

### ✅ COPY (Large Files)
- `saved_models/*.pkl` - Trained models (50-500MB each)
  - Only copy the models you want to deploy!

### ❌ DON'T COPY
- `data/` - Training datasets (too large, not needed)
- `scripts/` - Training scripts (not needed on VPS)
- `archive/` - Experiments (not needed)
- `results/` - Results (not needed)

---

## 🚀 Quick Start

```bash
cd /home/infres/billy-22/projets/esa_challenge_kaggle/deepfake-text-detector

# Deploy everything (first time)
./deploy_to_vps.sh user@your-vps-ip:/home/user/deepfake-backend

# Or manually:
rsync -avz --progress \
  --exclude 'data/' \
  --exclude 'archive/' \
  --exclude 'scripts/' \
  . user@your-vps-ip:/home/user/deepfake-backend/
```

Then deploy in Coolify dashboard!

---

## ❓ FAQ

**Q: Do I need to install dependencies on VPS?**
A: No! Docker handles it automatically.

**Q: Can I still train on local?**
A: YES! Keep training locally. Just sync new models to VPS.

**Q: What if VPS has no GPU?**
A: API will use CPU (slower but works). GPU recommended for production.

**Q: How do I update models?**
A: Train locally, then: `rsync saved_models/*.pkl user@vps:~/backend/saved_models/`

**Q: Do I need all my data on VPS?**
A: NO! VPS only needs API code + trained models. No raw data needed.

---

## 📞 Next Steps

1. Read: **[DEPLOYMENT_WORKFLOW.md](./DEPLOYMENT_WORKFLOW.md)** for full details
2. Test locally: `./start_api.sh`
3. Copy to VPS: `./deploy_to_vps.sh user@vps:/path`
4. Deploy in Coolify
5. Update frontend with API URL
6. Done! 🎉

---

**Bottom Line:** Keep your powerful training setup local, deploy lightweight API to VPS! 🚀
