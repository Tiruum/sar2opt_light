import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Добавляем корень проекта в sys.path
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))

from src.models.cfrwd.gen import CFRWDGenerator, CFRBranch, HFCFBranch, FinalDecoderBlock

def test_overfit(model_name, model, x, y, iterations=400, lr=5e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(x.device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(x.device)

    print(f"\n[{model_name}] Начинаем overfit-тест (один батч)...")
    init_loss = None
    for i in range(iterations):
        optimizer.zero_grad()
        out = model(x)
        
        fusion_w = None
        if isinstance(out, tuple):
            if len(out) == 2:
                out, fusion_w = out
            else:
                out = out[0]
            
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        val = loss.item()
        
        if i == 0 or (i+1) % 30 == 0:
            with torch.no_grad():
                psnr_val = psnr_fn(out, y).item()
                ssim_val = ssim_fn(out, y).item()
                
            msg = f"  Шаг {i+1:3d} | Loss: {val:.4f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.3f}"
            if fusion_w is not None:
                # W[0]: CFR, W[1]: HFCF
                w_cfr = fusion_w[:, 0].mean().item()
                w_hfcf = fusion_w[:, 1].mean().item()
                msg += f" | Mix: CFR {w_cfr:.2f}, HFCF {w_hfcf:.2f}"
            print(msg)
            
            if i == 0:
                init_loss = val
            
    success = val < init_loss * 0.1 # падение на 90%
    print(f"[{model_name}] Итог: {'УСПЕХ (обучается)' if success else 'ПРОВАЛ (не падает)'}")
    return success

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")
    
    from torch.utils.data import DataLoader
    from src.data.transforms import get_common_transform, get_input_specific, get_optical_specific, get_resize_transform

    dataset_names = ["sen12", "sen12_full", "qxslab"] # Можно менять на "sen12", "sen12_full", "qxslab"

    
    for dataset_name in dataset_names:
        print(f"Загрузка реальных данных из датасета {dataset_name.upper()} (4 картинки)...")
        
        if dataset_name == "sen12":
            from src.data.sen12.dataset import SEN12
            dataset = SEN12(
                root_dir=str(root_dir / "data" / "sen12"),
                common_transform=get_common_transform(),
                input_specific=get_input_specific(sar_channels=1),
                optical_specific=get_optical_specific(),
                resize_transform=get_resize_transform(256),
                sar_channels=1
            )
        elif dataset_name == "sen12_full":
            from src.data.sen12_full.dataset import SEN12Full
            dataset = SEN12Full(
                root_dir=str(root_dir / "data" / "sen12_full"),
                common_transform=get_common_transform(),
                input_specific=get_input_specific(sar_channels=1),
                optical_specific=get_optical_specific(),
                resize_transform=get_resize_transform(256),
                sar_channels=1
            )
        elif dataset_name == "qxslab":
            from src.data.qxslab.dataset import QXSLABDataset
            dataset = QXSLABDataset(
                root_dir=str(root_dir / "data" / "QXSLAB_SAROPT"),
                common_transform=get_common_transform(),
                input_specific=get_input_specific(sar_channels=1),
                optical_specific=get_optical_specific(),
                resize_transform=get_resize_transform(256),
                sar_channels=1
            )
        else:
            raise ValueError("Unknown dataset")
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        print(f"Загружен батч: x={x.shape}, y={y.shape} (min: {x.min():.2f}, max: {x.max():.2f})")
        
        # 1. Тест полной модели генератора
        gen = CFRWDGenerator(in_channels=1).to(device)
        test_overfit("Full CFRWDGenerator", gen, x, y)
        
        # 2. Тест только ветки CFR
        class CFRWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch = CFRBranch(in_channels=1)
                self.final = FinalDecoderBlock(32, 3)
            def forward(self, x):
                return torch.tanh(self.final(self.branch(x)))
                
        cfr_only = CFRWrapper().to(device)
        test_overfit("CFR Branch Only", cfr_only, x, y)
        
        # 3. Тест только ветки HFCF
        class HFCFWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch = HFCFBranch(in_channels=1)
                self.final = FinalDecoderBlock(32, 3)
            def forward(self, x):
                feats, _, _ = self.branch(x)
                return torch.tanh(self.final(feats))
                
        hfcf_only = HFCFWrapper().to(device)
        test_overfit("HFCF Branch Only", hfcf_only, x, y)
