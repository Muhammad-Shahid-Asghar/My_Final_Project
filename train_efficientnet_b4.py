# ============================================================
# SkinCare AI — EfficientNet-B4 Training
# Run: python train_efficientnet_b0.py
# ============================================================

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import timm, albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from pathlib import Path
import cv2, os, shutil, gc
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR   = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI\DataSet\Train"
VAL_DIR     = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI\DataSet\Val"
DERMAI_DIR  = r"C:\Users\M Shahid Asghar\Disease Dectection Project\DermAI"
PLOT_DIR    = os.path.join(DERMAI_DIR, "plots", "efficientnet_b4")

MODEL_NAME  = 'efficientnet_b4'
SAVE_NAME   = 'best_efficientnet_b4.pth'
NUM_CLASSES = 11  # 10 diseases + Normal Skin
IMG_SIZE    = 224
BATCH_SIZE  = 4
GRAD_ACCUM  = 4
EPOCHS      = 30
NUM_WORKERS = 0
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    "Eczema", "Warts / Molluscum", "Melanoma", "Atopic Dermatitis",
    "Basal Cell Carcinoma", "Melanocytic Nevi", "Benign Keratosis",
    "Psoriasis / Lichen", "Seborrheic Keratoses", "Tinea / Ringworm",
    "Normal Skin",
]
CLASS_SHORT = ["Eczema","Warts","Melanoma","Atopic D.","BCC","Nevi","Ben.Ker","Psoriasis","Seb.Ker","Tinea","Normal Skin"]

os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(DERMAI_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print(f"\n{'='*60}")
print(f"  SkinCare AI — EfficientNet-B4 Training")
print(f"{'='*60}")
print(f"  Batch      : {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
print(f"  Device     : {DEVICE}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : {props.total_memory/1e9:.1f} GB")
print(f"  Save As    : {os.path.join(DERMAI_DIR, SAVE_NAME)}")
print(f"{'='*60}\n")

# ============================================================
# AUGMENTATION
# ============================================================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
    A.GaussianBlur(blur_limit=(3,5), p=0.1),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# ============================================================
# DATASET
# ============================================================
class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        found_folders = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        print(f"\nClasses in: {root_dir}")
        for idx, folder_name in enumerate(found_folders):
            folder_path = self.root_dir / folder_name
            count = 0
            for ext in ['*.jpg','*.jpeg','*.png','*.bmp']:
                for img_path in folder_path.glob(ext):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
                    count += 1
            print(f"  [{idx:2d}] {folder_name[:40]:<40} ({count:5d})")
        print(f"  Total: {len(self.image_paths)}\n")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, torch.tensor(self.labels[idx], dtype=torch.long)

# ============================================================
# CUTMIX
# ============================================================
def cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B = images.size(0)
    rand_index = torch.randperm(B).to(images.device)
    labels_a = labels
    labels_b = labels[rand_index]
    W, H = images.size(2), images.size(3)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w//2, 0); x2 = min(cx + cut_w//2, W)
    y1 = max(cy - cut_h//2, 0); y2 = min(cy + cut_h//2, H)
    mixed = images.clone()
    mixed[:, :, x1:x2, y1:y2] = images[rand_index, :, x1:x2, y1:y2]
    lam = 1 - ((x2-x1)*(y2-y1)/(W*H))
    return mixed, labels_a, labels_b, lam

# ============================================================
# MODEL
# ============================================================
class SkinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            try: self.backbone.set_grad_checkpointing(enable=True)
            except: pass
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(self.backbone.num_features, NUM_CLASSES)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        print(f"  Backbone  : {MODEL_NAME}")
        print(f"  Features  : {self.backbone.num_features}")

    def forward(self, x):
        return self.classifier(self.dropout(self.backbone(x)))

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

# ============================================================
# SAMPLER
# ============================================================
def get_sampler(dataset):
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_w = class_weights[labels]
    actual_size = len(dataset)
    print(f"  Sampler: {actual_size} samples ({actual_size//BATCH_SIZE} batches/epoch)")
    return WeightedRandomSampler(
        weights=torch.FloatTensor(sample_w),
        num_samples=actual_size,
        replacement=True
    )

# ============================================================
# SAVE MODEL
# ============================================================
def save_model(model, epoch, val_auc, val_acc, optimizer):
    checkpoint = {
        'epoch': epoch, 'model_name': MODEL_NAME,
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'auc': val_auc, 'accuracy': val_acc,
        'num_classes': NUM_CLASSES, 'img_size': IMG_SIZE,
        'class_names': CLASS_NAMES,
    }
    torch.save(checkpoint, SAVE_NAME)
    dermai_path = os.path.join(DERMAI_DIR, SAVE_NAME)
    try:
        shutil.copy2(SAVE_NAME, dermai_path)
        size_mb = os.path.getsize(dermai_path) / (1024*1024)
        print(f"  [SAVED] AUC:{val_auc:.4f} Acc:{val_acc:.2f}% → {dermai_path} ({size_mb:.1f}MB)")
    except Exception as e:
        print(f"  [SAVED] AUC:{val_auc:.4f} (local only — {e})")

# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, use_amp):
    model.train()
    total_loss = 0.0; correct = 0; total = 0; nan_count = 0; oom_count = 0
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Train", ncols=90)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        use_cutmix = (np.random.random() < 0.3)
        la = lb = lam = None
        try:
            if use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    if use_cutmix:
                        images, la, lb, lam = cutmix(images, labels)
                        out = model(images)
                        loss = lam*criterion(out,la) + (1-lam)*criterion(out,lb)
                    else:
                        out = model(images); loss = criterion(out, labels)
                    loss = loss / GRAD_ACCUM
                scaler.scale(loss).backward()
                if (step+1) % GRAD_ACCUM == 0 or (step+1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                if use_cutmix:
                    images, la, lb, lam = cutmix(images, labels)
                    out = model(images)
                    loss = lam*criterion(out,la) + (1-lam)*criterion(out,lb)
                else:
                    out = model(images); loss = criterion(out, labels)
                loss = loss / GRAD_ACCUM
                loss.backward()
                if (step+1) % GRAD_ACCUM == 0 or (step+1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad()
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1; optimizer.zero_grad(); continue
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect(); optimizer.zero_grad()
            oom_count += 1; pbar.set_postfix({'OOM': oom_count}); continue
        total_loss += loss.item() * GRAD_ACCUM
        _, pred = out.max(1)
        target = la if use_cutmix else labels
        correct += pred.eq(target).sum().item(); total += labels.size(0)
        pbar.set_postfix({'loss': f'{loss.item()*GRAD_ACCUM:.3f}', 'acc': f'{100.*correct/max(total,1):.1f}%'})
    if nan_count > 0: print(f"  WARNING: {nan_count} NaN batches")
    if oom_count > 0: print(f"  WARNING: {oom_count} OOM batches")
    return total_loss / max(len(loader)-nan_count-oom_count, 1), 100.*correct/max(total,1)

# ============================================================
# VALIDATE
# ============================================================
def validate(model, loader, criterion):
    model.eval(); total_loss = 0.0; all_preds = []; all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val  ", ncols=90):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            try:
                outputs = model(images)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            loss = criterion(outputs, labels)
            if torch.isnan(loss): continue
            total_loss += loss.item()
            probs = torch.softmax(outputs.float(), dim=1)
            if not torch.isnan(probs).any():
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    if not all_preds: return 0.,0.,0.,None,None
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    if np.isnan(all_preds).any(): return 0.,0.,0.,None,None
    bin_labels = label_binarize(all_labels, classes=range(NUM_CLASSES))
    try:
        auc_score = roc_auc_score(bin_labels, all_preds, average='macro', multi_class='ovr')
    except ValueError as e:
        print(f"  WARNING: AUC failed — {e}"); auc_score = 0.
    accuracy = (all_preds.argmax(1) == all_labels).mean() * 100
    return total_loss/max(len(loader),1), accuracy, auc_score, all_preds, all_labels

# ============================================================
# PLOTS
# ============================================================
def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"SkinCare AI — {MODEL_NAME}", fontsize=15, fontweight='bold')
    ep = range(1, len(history['train_loss'])+1)
    axes[0].plot(ep, history['train_loss'], '#2196F3', lw=2, marker='o', ms=3, label='Train')
    axes[0].plot(ep, history['val_loss'],   '#F44336', lw=2, marker='s', ms=3, label='Val')
    axes[0].set_title('Loss', fontweight='bold'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, history['train_acc'], '#2196F3', lw=2, marker='o', ms=3, label='Train')
    axes[1].plot(ep, history['val_acc'],   '#F44336', lw=2, marker='s', ms=3, label='Val')
    axes[1].axvline(x=history['best_epoch'], color='green', ls='--', lw=1.5, label=f"Best Ep{history['best_epoch']}")
    axes[1].set_title('Accuracy', fontweight='bold'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ep, history['val_auc'], '#9C27B0', lw=2, marker='^', ms=3, label='AUC-ROC')
    best_a = max(history['val_auc'])
    axes[2].axhline(y=best_a, color='green', ls='--', lw=1.5, label=f"Best={best_a:.4f}")
    axes[2].set_title('AUC-ROC', fontweight='bold'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([max(0.5, min(history['val_auc'])-0.05), 1.0])
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [PLOT] {save_path}")

def plot_confusion_matrix(all_labels, all_preds_cls, save_path):
    cm = confusion_matrix(all_labels, all_preds_cls)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(f"SkinCare AI — {MODEL_NAME} Confusion Matrix", fontsize=15, fontweight='bold')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, ax=axes[0])
    axes[0].set_title('Count', fontweight='bold'); axes[0].tick_params(axis='x', rotation=45)
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title('Percentage (%)', fontweight='bold'); axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [PLOT] {save_path}")

def plot_per_class_auc(all_labels, all_preds_prob, save_path):
    bin_labels = label_binarize(all_labels, classes=range(NUM_CLASSES))
    aucs = []
    for i in range(NUM_CLASSES):
        try: a = roc_auc_score(bin_labels[:,i], all_preds_prob[:,i])
        except: a = 0.
        aucs.append(a)
    colors = ['#4CAF50' if a>=0.9 else '#FF9800' if a>=0.8 else '#F44336' for a in aucs]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CLASS_SHORT, aucs, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0.9, color='green', ls='--', lw=1.5, label='0.90 target')
    ax.axhline(y=np.mean(aucs), color='blue', ls='--', lw=1.5, label=f'Mean={np.mean(aucs):.4f}')
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.003, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(f"SkinCare AI — {MODEL_NAME} Per-Class AUC", fontweight='bold', fontsize=13)
    ax.set_ylabel('AUC-ROC'); ax.set_ylim([0.5, 1.05]); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [PLOT] {save_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    train_dataset = SkinDataset(TRAIN_DIR, transform=train_transform)
    val_dataset   = SkinDataset(VAL_DIR,   transform=val_transform)
    assert len(train_dataset) > 0, f"No training images found in {TRAIN_DIR}"
    assert len(val_dataset)   > 0, f"No val images found in {VAL_DIR}"

    sampler      = get_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE*4, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model    = SkinModel().to(DEVICE)
    scaler   = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    use_amp  = torch.cuda.is_available()
    criterion = nn.CrossEntropyLoss()

    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_auc':[], 'best_epoch':1}
    best_auc = 0.0; best_epoch = 0; best_preds_prob = None; best_preds_cls = None; best_labels_arr = None

    print(f"\n  Phase 1: Epochs 1-5   (classifier head)")
    print(f"  Phase 2: Epochs 6-{EPOCHS}  (full network)")
    print("-"*60)

    for epoch in range(EPOCHS):
        if epoch == 5:
            print("\n>>> Phase 2: Full Network Unfrozen <<<")
            model.unfreeze_backbone()
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(),   'lr': 1e-4},
                {'params': model.classifier.parameters(), 'lr': 3e-4}
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-5, eta_min=1e-7)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1:3d}/{EPOCHS}]  LR:{current_lr:.2e}  AMP:{use_amp}")
        if torch.cuda.is_available():
            print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB alloc / {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, use_amp)
        val_loss, val_acc, val_auc, preds_prob, preds_labels = validate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss);     history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"  Train → Loss:{train_loss:.4f}  Acc:{train_acc:.2f}%")
        print(f"  Val   → Loss:{val_loss:.4f}  Acc:{val_acc:.2f}%  AUC:{val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc; best_epoch = epoch+1
            best_preds_prob = preds_prob
            best_preds_cls  = preds_prob.argmax(1) if preds_prob is not None else None
            best_labels_arr = preds_labels
            history['best_epoch'] = best_epoch
            save_model(model, epoch+1, val_auc, val_acc, optimizer)

        if (epoch+1) % 5 == 0 or epoch == EPOCHS-1:
            plot_training_history(history, os.path.join(PLOT_DIR, f"{MODEL_NAME}_history_ep{epoch+1}.png"))

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Final plots
    plot_training_history(history, os.path.join(PLOT_DIR, f"{MODEL_NAME}_history_FINAL.png"))
    if best_preds_prob is not None and best_labels_arr is not None:
        plot_confusion_matrix(best_labels_arr, best_preds_cls, os.path.join(PLOT_DIR, f"{MODEL_NAME}_confusion_matrix.png"))
        plot_per_class_auc(best_labels_arr, best_preds_prob, os.path.join(PLOT_DIR, f"{MODEL_NAME}_per_class_auc.png"))
        print(f"\n--- Classification Report ---")
        print(classification_report(best_labels_arr, best_preds_cls, target_names=CLASS_NAMES, digits=4))

    print(f"\n{'='*60}")
    print(f"  EfficientNet-B4 Training Complete!")
    print(f"  Best AUC   : {best_auc:.4f}")
    print(f"  Best Epoch : {best_epoch}")
    print(f"  Model      : {os.path.join(DERMAI_DIR, SAVE_NAME)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
