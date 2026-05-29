import os
import time
import json
import logging
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from mp.complex import ComplexBatch
from net.cwn_abcort_transformer_v2 import CWNABCoRTTransformerV2
from mp.smrt_dataset import SMRTComplexDataset 

class TorchScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x_tensor):
        self.mean = x_tensor.mean(dim=0, keepdim=True)
        self.std = x_tensor.std(dim=0, unbiased=False, keepdim=True)
        self.std[self.std == 0] = 1.0

    def transform(self, x_tensor):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (x_tensor - self.mean.to(x_tensor.device)) / self.std.to(x_tensor.device)

    def inverse_transform(self, x_tensor):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return x_tensor * self.std.to(x_tensor.device) + self.mean.to(x_tensor.device)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0): # [优化] 增加耐受度到 50
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ================= 配置 (SOTA 调优版) =================

# ================= 配置 (SOTA 调优版) =================

class Config:
    # 🌟 1. 独立指定官方训练集和测试集的路径
    TRAIN_CSV_PATH = os.getenv('TRAIN_CSV_PATH', 'data/SMRT_train.csv') 
    TEST_CSV_PATH = os.getenv('TEST_CSV_PATH', 'data/SMRT_test.csv')
    
    # 🌟 2. 使用两个不同的文件夹存特征缓存，彻底防止图数据冲突
    TRAIN_DATA_ROOT = os.getenv('TRAIN_DATA_ROOT', 'smrt_cwn_data_train')
    TEST_DATA_ROOT = os.getenv('TEST_DATA_ROOT', 'smrt_cwn_data_test')
    
    RESULT_DIR = os.getenv('RESULT_DIR', 'results_CWN_ABCoRT_Transformer') 
    CHECKPOINT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
    PLOT_DIR = os.path.join(RESULT_DIR, 'plots')
    LOG_FILE = os.path.join(RESULT_DIR, 'train.log')
    TEST_RES_FILE = os.path.join(RESULT_DIR, 'test_predictions.csv')
    
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '64'))        
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-4')) 
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', '1e-2'))
    EPOCHS = int(os.getenv('EPOCHS', '150'))            
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MAX_RING_SIZE = int(os.getenv('MAX_RING_SIZE', '6')) 
    USE_EDGE_ATTR = True
    HUBER_BETA = 1.0

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

logging.basicConfig(filename=Config.LOG_FILE, level=logging.INFO, format='%(message)s', filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def log_json(data):
    logging.info(json.dumps(data, ensure_ascii=False))

def complex_collate_fn(batch):
    return ComplexBatch.from_complex_list(batch)

def load_valid_smiles(csv_path):
    from rdkit import Chem

    df = pd.read_csv(csv_path, engine="python")
    df.columns = [str(c).lower().strip() for c in df.columns]

    if 'smile' in df.columns and 'smiles' not in df.columns:
        df.rename(columns={'smile': 'smiles'}, inplace=True)

    if 'smiles' not in df.columns or 'rt' not in df.columns:
        df = pd.read_csv(csv_path, sep=r"\s+", names=["smiles", "rt"], header=0, engine="python")

    df = df[df['rt'] > 300.0].copy()

    valid_smiles = []
    valid_rt = []

    for _, row in df.iterrows():
        smiles = row.get("smiles", None)
        rt = row.get("rt", None)

        if pd.isna(smiles):
            continue

        try:
            rt = float(rt)
        except:
            continue

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue

        valid_smiles.append(str(smiles))
        valid_rt.append(rt)

    return valid_smiles, valid_rt

@torch.no_grad()
def export_predictions(model, loader, smiles_list, save_path):
    model.eval()

    preds_list = []
    targets_list = []

    for batch in loader:
        batch = batch.to(Config.DEVICE)
        targets = batch.y.view(-1)
        pred = model(batch)

        if isinstance(pred, tuple):
            pred = pred[0]

        preds_list.append(pred.view(-1).cpu())
        targets_list.append(targets.cpu())

    preds = torch.cat(preds_list).numpy()
    targets = torch.cat(targets_list).numpy()

    n = min(len(smiles_list), len(preds), len(targets))

    df_out = pd.DataFrame({
        "SMILES": smiles_list[:n],
        "Actual_RT": targets[:n],
        "Predicted_RT": preds[:n],
        "Abs_Error": np.abs(targets[:n] - preds[:n])
    })

    df_out.to_csv(save_path, index=False)
    print(f"✅ 已导出预测文件: {save_path}")

def fit_scalers(train_loader):
    all_ys = []
    for batch in train_loader:
        all_ys.append(batch.y)
    y_tensor = torch.cat(all_ys, dim=0).view(-1, 1).float()
    y_scaler = TorchScaler()
    y_scaler.fit(y_tensor)
    return y_scaler

# ================= 训练与评估核心 =================

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_mae_raw = 0
    steps = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    
    for batch in pbar:
        batch = batch.to(Config.DEVICE)
        # scaled_y = y_scaler.transform(batch.y.view(-1, 1)).view(-1)
        # 🌟 直接使用最原始的保留时间(秒)，不缩放！
        targets = batch.y.view(-1)
        optimizer.zero_grad()
        pred = model(batch)
        if isinstance(pred, tuple): pred = pred[0]
        
        # 使用优化后的 Huber Loss
        loss = F.smooth_l1_loss(pred.view(-1), targets, beta=Config.HUBER_BETA)
        # loss = F.l1_loss(pred.view(-1), scaled_y)
        loss.backward()
        
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        mae_raw = F.l1_loss(pred.view(-1), targets).item()
        total_loss += loss.item()
        total_mae_raw += mae_raw
        steps += 1
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'MAE': f"{mae_raw:.1f}s"})
        
    return total_loss / steps, total_mae_raw / steps

@torch.no_grad()
def evaluate(model, loader, epoch, prefix='val'):
    model.eval()
    preds_list, targets_list = [], []
    total_loss = 0
    steps = 0
    
    for batch in loader:
        batch = batch.to(Config.DEVICE)
        # 🌟 1. 验证集也必须用真实的秒数！
        targets = batch.y.view(-1)
        pred = model(batch)
        if isinstance(pred, tuple): pred = pred[0]
        
        loss = F.smooth_l1_loss(pred.view(-1), targets, beta=Config.HUBER_BETA)
        # loss = F.l1_loss(pred.view(-1), scaled_y)
        total_loss += loss.item()
        
        # 🌟 3. 直接存入列表，不需要任何 inverse_transform
        preds_list.append(pred.view(-1).cpu())       
        targets_list.append(targets.cpu())  
        steps += 1
        
    preds = torch.cat(preds_list).numpy()
    targets = torch.cat(targets_list).numpy()
    
    # 计算 image_6cd4d6.png 上的全套指标
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    medae = median_absolute_error(targets, preds)
    mean_bias = np.mean(preds - targets) # 平均偏差
    
    # [新增] 计算 MRE (%) 和 MedRE (%)，加上 1e-8 防止分母为 0
    relative_errors = np.abs(targets - preds) / (np.abs(targets) + 1e-8)
    mre = np.mean(relative_errors) * 100
    medre = np.median(relative_errors) * 100
    
    metrics = {
        f'{prefix}_loss': total_loss / steps, 
        f'{prefix}_mae': f"{mae:.4f}",        
        f'{prefix}_rmse': f"{rmse:.4f}",
        f'{prefix}_r2': f"{r2:.4f}",
        f'{prefix}_medae': f"{medae:.4f}",
        f'{prefix}_mre': f"{mre:.4f}",       # [新增]
        f'{prefix}_medre': f"{medre:.4f}",   # [新增]
        f'{prefix}_bias': f"{mean_bias:.4f}",
        'epoch': epoch
    }
    log_json(metrics)
    
    return float(metrics[f'{prefix}_loss']), float(metrics[f'{prefix}_mae']), float(metrics[f'{prefix}_r2']), float(metrics[f'{prefix}_mre']), float(metrics[f'{prefix}_medae']), float(metrics[f'{prefix}_medre']), targets, preds

def plot_final_result(targets, preds, save_path, history=None):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8])

    # 1. 散点图
    ax_scatter = fig.add_subplot(gs[0, 0])
    errors = np.abs(targets - preds)
    sc = ax_scatter.scatter(targets, preds, c=errors, cmap='viridis_r', alpha=0.6, s=40)
    plt.colorbar(sc, ax=ax_scatter, label='Abs Error (s)')
    ax_scatter.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    ax_scatter.set_title('Actual vs Predicted RT', fontsize=15)

    # 2. 误差分布 (对应 image_6cd4d6.png 的右侧直方图)
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.hist(preds - targets, bins=60, color='skyblue', edgecolor='black')
    ax_hist.axvline(np.mean(preds - targets), color='red', linestyle='--')
    ax_hist.set_title('Error Distribution', fontsize=15)

    # 3. 训练进度 (实时监控 Train vs Val MAE)
    if history:
        ax_curve = fig.add_subplot(gs[1, :])
        ax_curve.plot(history['train_mae'], label='Train MAE (s)', linewidth=2)
        ax_curve.plot(history['val_mae'], label='Val MAE (s)', linewidth=2)
        ax_curve.set_title('Overfitting Monitor: Train vs Val MAE', fontsize=15)
        ax_curve.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= 主程序 =================

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # # === 🚀 1. 动态获取特征后缀，清空训练和测试的旧缓存 ===
    # suffix = f"_r{Config.MAX_RING_SIZE}_Full46D_Embedded"
    # if Config.USE_EDGE_ATTR: 
    #     suffix += "_E"
        
    # train_cache_dir = os.path.join(Config.TRAIN_DATA_ROOT, 'processed' + suffix)
    # test_cache_dir = os.path.join(Config.TEST_DATA_ROOT, 'processed' + suffix)
    
    # for c_dir in [train_cache_dir, test_cache_dir]:
    #     if os.path.exists(c_dir):
    #         import shutil
    #         shutil.rmtree(c_dir)
    #         print(f"🧹 已成功清空旧特征缓存: {c_dir}")

    # === 📊 2. 加载训练集，并按 90:10 划分为 Train 和 Val ===
    print("\n=== 加载并划分训练集 ===")
    dataset_train_full = SMRTComplexDataset(Config.TRAIN_DATA_ROOT, Config.TRAIN_CSV_PATH, Config.MAX_RING_SIZE, use_edge_features=Config.USE_EDGE_ATTR)
    
    total_train_len = len(dataset_train_full)
    train_len = int(0.9 * total_train_len)
    val_len = total_train_len - train_len 
    
    train_set, val_set = random_split(
        dataset_train_full, 
        [train_len, val_len], 
        generator=torch.Generator().manual_seed(seed)
    )

    train_valid_smiles, _ = load_valid_smiles(Config.TRAIN_CSV_PATH)
    test_valid_smiles, _ = load_valid_smiles(Config.TEST_CSV_PATH)

    train_smiles = [train_valid_smiles[i] for i in train_set.indices]
    val_smiles = [train_valid_smiles[i] for i in val_set.indices]

    # === 🎯 3. 独立加载官方测试集 (完全不参与随机切分) ===
    print("\n=== 加载独立测试集 ===")
    test_set = SMRTComplexDataset(Config.TEST_DATA_ROOT, Config.TEST_CSV_PATH, Config.MAX_RING_SIZE, use_edge_features=Config.USE_EDGE_ATTR)
    
    # 注意：如果训练卡死，把这里的 num_workers 改为 0
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=complex_collate_fn, num_workers=4)
    train_export_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=complex_collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=complex_collate_fn)
    test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=complex_collate_fn)
    
    # --- 下方的模型初始化与训练循环保持原样不变 ---
    model = CWNABCoRTTransformerV2(
        out_size=1,
        num_layers=int(os.getenv('CWN_LAYERS', '6')),
        hidden=int(os.getenv('CWN_HIDDEN', '256')),
        d_model=int(os.getenv('TRANS_DMODEL', '256')),
        transformer_layers=int(os.getenv('TRANS_LAYERS', '2')),
        transformer_heads=int(os.getenv('TRANS_HEADS', '8')),
        transformer_dropout=float(os.getenv('TRANS_DROPOUT', '0.10')),
        dropout_rate=float(os.getenv('CWN_DROPOUT', '0.0')),
        max_dim=2,
        jump_mode='cat',
        use_coboundaries=True,
    ).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, amsgrad=True, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
    early_stopping = EarlyStopping(patience=30)

    history = {'train_mae': [], 'val_mae': []}
    best_mae = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        # 获取 Train MAE
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, epoch)
        # 获取 Val MAE
        val_loss, val_mae, val_r2, _, _, _, _, _ = evaluate(model, val_loader, epoch, prefix='val')
        
        scheduler.step()
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # 实时打印监控 (过拟合检查点)
        print(f"Epoch {epoch:03d} | Train MAE: {train_mae:.1f}s | Val MAE: {val_mae:.1f}s | R2: {val_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
            
        early_stopping(val_mae)
        if early_stopping.early_stop: break

    # 最终测试并绘图
    model.load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')))
    # [修改 1] 使用 8 个变量接住 evaluate 返回的所有指标，并补上 y_scaler 参数
    _, test_mae, test_r2, test_mre, test_medae, test_medre, t_y, t_p = evaluate(model, test_loader, -1, prefix='test')

    print("\n=== 导出 base CWN train/val/test 预测，供双视角融合使用 ===")

    export_predictions(
        model,
        train_export_loader,
        train_smiles,
        os.path.join(Config.RESULT_DIR, "base_train_predictions.csv")
    )

    export_predictions(
        model,
        val_loader,
        val_smiles,
        os.path.join(Config.RESULT_DIR, "base_val_predictions.csv")
    )

    export_predictions(
        model,
        test_loader,
        test_valid_smiles,
        os.path.join(Config.RESULT_DIR, "base_test_predictions.csv")
    )
    
    plot_final_result(t_y, t_p, os.path.join(Config.PLOT_DIR, 'Final_Dashboard.png'), history=history)
    
    # [修改 2] 打印完美对齐 ABCoRT 论文表格的输出格式
    print("\n" + "="*50)
    print(f"🥇 Final TopoRT-Net Test Results (vs ABCoRT):")
    print(f"MAE:   {test_mae:.2f} s")
    print(f"MRE:   {test_mre:.2f} %")
    print(f"MedAE: {test_medae:.2f} s")
    print(f"MedRE: {test_medre:.2f} %")
    print(f"R2:    {test_r2:.4f}")
    print("="*50 + "\n")
    # ==========================================================
    # 🚀 新增模块：找出误差最大的“十字幽灵”分子并可视化
    # ==========================================================
    print("\n=== 开始提取并绘制误差最大的分子 ===")
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # 1. 重新读取测试集以获取对应的 SMILES (保持和 dataset 里严格一致的过滤逻辑以对齐数据)
    df_test = pd.read_csv(Config.TEST_CSV_PATH, engine="python")
    df_test.columns = [str(c).lower().strip() for c in df_test.columns]
    if 'smile' in df_test.columns and 'smiles' not in df_test.columns:
        df_test.rename(columns={'smile': 'smiles'}, inplace=True)
    if 'smiles' not in df_test.columns or 'rt' not in df_test.columns:
        df_test = pd.read_csv(Config.TEST_CSV_PATH, sep=r"\s+", names=["smiles", "rt"], header=0, engine="python")
    
    valid_smiles = []
    for idx, row in df_test.iterrows():
        smiles = row.get("smiles", None)
        rt = row.get("rt", None)
        if pd.isna(smiles): continue
        try:
            rt = float(rt)
        except: continue
        if rt <= 300.0: continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None: continue
        valid_smiles.append(str(smiles))
    
    # 2. 构建 DataFrame 进行对齐和误差计算
    # t_y 和 t_p 是我们在 evaluate 函数里拿到的真实的 targets 和 preds
    df_res = pd.DataFrame({
        'SMILES': valid_smiles,
        'Actual_RT': t_y,
        'Predicted_RT': t_p,
        'Abs_Error': np.abs(t_y - t_p)
    })
    
    # 3. 按绝对误差降序排列，并把完整结果保存到 CSV，方便你随时查阅
    df_res = df_res.sort_values(by='Abs_Error', ascending=False).reset_index(drop=True)
    csv_out_path = os.path.join(Config.RESULT_DIR, 'worst_predictions_analysis.csv')
    df_res.to_csv(csv_out_path, index=False)
    print(f"📄 完整的测试集误差分析表已保存至: {csv_out_path}")
    
    # 4. 画出 Top 12 误差最大的分子 2D 结构图
    top_k = 12
    mols = [Chem.MolFromSmiles(s) for s in df_res['SMILES'].head(top_k)]
    legends = [f"Act: {row['Actual_RT']:.1f}s\nPred: {row['Predicted_RT']:.1f}s\nErr: {row['Abs_Error']:.1f}s" 
               for _, row in df_res.head(top_k).iterrows()]
               
    # 如果分子画图报错，把 subImgSize 稍微改大一点
    try:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(350, 350), legends=legends)
        img_out_path = os.path.join(Config.PLOT_DIR, 'Worst_Predictions_Top12.png')
        img.save(img_out_path)
        print(f"🖼️ 成功！误差最大的分子可视化图已保存至: {img_out_path}")
    except Exception as e:
        print(f"⚠️ 画图时出现小错误: {e}")
if __name__ == "__main__":
    main()