import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.NSST import NSST, Conf

def symext_pad(x: torch.Tensor, p: int, q: int, shift: list) -> torch.Tensor:
    m, n = x.shape[-2:]
    s1, s2 = int(shift[0]), int(shift[1])

    ss = int(p // 2) - s1 + 1
    rr = int(q // 2) - s2 + 1

    left = torch.flip(x[..., :ss], dims=[-1])
    right_start = max(0, n - (p + s1))
    right = torch.flip(x[..., right_start:], dims=[-1])
    
    yT = torch.cat([left, x, right], dim=-1)
    
    top = torch.flip(yT[..., :rr, :], dims=[-2])
    bottom_start = max(0, m - (q + s2))
    bottom = yT[..., bottom_start:, :]
    
    yT2 = torch.cat([top, yT, bottom], dim=-2)
    
    return yT2[..., :m+p-1, :n+q-1]

class NSSTDecomposer(nn.Module):
    def __init__(self):
        super().__init__()
        nsst_np = NSST()
        nsst_np.atrous_filters_init()
        nsst_np.shearing_filters_myer()
        
        self.dcompSize = Conf.sp.dcompSize
        self.dcomp = Conf.sp.dcomp
        self.dsize = Conf.sp.dsize
        
        for i, f in enumerate(nsst_np.atrous_filters):
            f_torch = torch.from_numpy(f.copy()).unsqueeze(0).unsqueeze(0).float()
            f_torch = torch.flip(f_torch, dims=[-1, -2])
            self.register_buffer(f'atrous_filter_{i}', f_torch)
            
        self.shear_shapes = []
        for i, level_filters in enumerate(nsst_np.shearing_filters):
            for k, f in enumerate(level_filters):
                f_torch = torch.from_numpy(f.copy()).unsqueeze(0).unsqueeze(0).float()
                f_torch = torch.flip(f_torch, dims=[-1, -2])
                self.register_buffer(f'shear_filter_{i}_{k}', f_torch)
            self.shear_shapes.append((len(level_filters), level_filters[0].shape))

    def forward(self, x):
        level = self.dcompSize
        y_atrous = self.atrous_dec(x, level)
        
        dst = [y_atrous[0]]
        
        for i in range(level):
            size = int(2 ** self.dcomp[i])
            level_dst = []
            for k in range(size):
                shear_f = getattr(self, f'shear_filter_{i}_{k}') * np.sqrt(self.dsize[i])
                
                kh, kw = shear_f.shape[-2:]
                pad_top = kh // 2
                pad_bottom = kh - 1 - pad_top
                pad_left = kw // 2
                pad_right = kw - 1 - pad_left
                
                x_pad = F.pad(y_atrous[i+1], (pad_left, pad_right, pad_top, pad_bottom), mode='circular')
                temp = F.conv2d(x_pad, shear_f)
                
                temp = torch.roll(temp, shifts=(-1, -1), dims=(-2, -1))
                level_dst.append(temp)
            dst.append(level_dst)
            
        return dst

    def atrous_dec(self, image, level):
        y0_filt = getattr(self, 'atrous_filter_1')
        y1_filt = getattr(self, 'atrous_filter_3')
        
        y = [None] * (level + 1)
        
        shift = [1.0, 1.0]
        sym0 = symext_pad(image, y0_filt.shape[-2], y0_filt.shape[-1], shift)
        img_low = F.conv2d(sym0, y0_filt)
        
        sym1 = symext_pad(image, y1_filt.shape[-2], y1_filt.shape[-1], shift)
        img_high = F.conv2d(sym1, y1_filt)
        
        y[level] = img_high
        curr_image = img_low

        for i in range(1, level):
            shift = [2.0 - 2**(i-1), 2.0 - 2**(i-1)]
            upsampling_factor = 2**i
            
            p0 = y0_filt.shape[-2] * upsampling_factor
            q0 = y0_filt.shape[-1] * upsampling_factor
            sym0 = symext_pad(curr_image, p0, q0, shift)
            out0 = F.conv2d(sym0, y0_filt, dilation=upsampling_factor)
            img_low = out0[..., upsampling_factor-1:, upsampling_factor-1:]
            
            p1 = y1_filt.shape[-2] * upsampling_factor
            q1 = y1_filt.shape[-1] * upsampling_factor
            sym1 = symext_pad(curr_image, p1, q1, shift)
            out1 = F.conv2d(sym1, y1_filt, dilation=upsampling_factor)
            img_high = out1[..., upsampling_factor-1:, upsampling_factor-1:]
            
            y[level - i] = img_high
            curr_image = img_low

        y[0] = curr_image
        return y

class NSSTReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        nsst_np = NSST()
        nsst_np.atrous_filters_init()
        
        for i, f in enumerate(nsst_np.atrous_filters):
            f_torch = torch.from_numpy(f.copy()).unsqueeze(0).unsqueeze(0).float()
            f_torch = torch.flip(f_torch, dims=[-1, -2])
            self.register_buffer(f'atrous_filter_{i}', f_torch)

    def forward(self, dst):
        level = len(dst) - 1
        y = [dst[0]]
        
        for i in range(1, level + 1):
            y.append(sum(dst[i]))
            
        return self.atrous_rec(y)

    def atrous_rec(self, y):
        NLevels = len(y) - 1
        x = y[0].clone()
        
        g0_filt = getattr(self, 'atrous_filter_0')
        g1_filt = getattr(self, 'atrous_filter_2')
        
        temp_p = x
        for i in range(NLevels - 1, 0, -1):
            shift = [2.0 - 2**(i-1), 2.0 - 2**(i-1)]
            upsampling_factor = 2**i
            
            p0 = g0_filt.shape[-2] * upsampling_factor
            q0 = g0_filt.shape[-1] * upsampling_factor
            s1 = symext_pad(temp_p, p0, q0, shift)
            out1 = F.conv2d(s1, g0_filt, dilation=upsampling_factor)
            c1 = out1[..., upsampling_factor-1:, upsampling_factor-1:]
            
            p1 = g1_filt.shape[-2] * upsampling_factor
            q1 = g1_filt.shape[-1] * upsampling_factor
            s2 = symext_pad(y[NLevels - i], p1, q1, shift)
            out2 = F.conv2d(s2, g1_filt, dilation=upsampling_factor)
            c2 = out2[..., upsampling_factor-1:, upsampling_factor-1:]
            
            temp_p = c1 + c2
            
        shift = [1.0, 1.0]
        s1 = symext_pad(temp_p, g0_filt.shape[-2], g0_filt.shape[-1], shift)
        c1 = F.conv2d(s1, g0_filt)
        
        s2 = symext_pad(y[NLevels], g1_filt.shape[-2], g1_filt.shape[-1], shift)
        c2 = F.conv2d(s2, g1_filt)
        
        return c1 + c2

class ResBlock(nn.Module):
    """2-conv residual block with GroupNorm + GELU."""
    def __init__(self, ch: int, num_groups: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(min(num_groups, ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class NSSTScaleProcessor(nn.Module):
    """
    Per-scale learnable SAR→OPT shearlet coefficient translator.
    in_ch: directional subbands at this scale (1=low-pass, 8=L1/L2, 16=L3/L4)
    out_ch = in_ch * 3  (3-channel optical expansion)
    apply_speckle_gate: True for directional subbands (speckle discrimination), False for low-pass
    """
    def __init__(self, in_ch: int, apply_speckle_gate: bool = True):
        super().__init__()
        from src.models.cfrwd.gen import SpeckleAwareModule, CBAM
        out_ch = in_ch * 3
        self.proj_in     = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.speckle_gate = SpeckleAwareModule(out_ch) if apply_speckle_gate else nn.Identity()
        self.cbam        = CBAM(out_ch)
        self.res1        = ResBlock(out_ch)
        self.res2        = ResBlock(out_ch)
        self.proj_out    = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        nn.init.normal_(self.proj_out.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        gate_out = self.speckle_gate(x)
        x = gate_out[0] if isinstance(gate_out, tuple) else gate_out
        x = self.cbam(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.proj_out(x)


class NSSTBranch(nn.Module):
    """
    NSST encoder-decoder branch: fixed decompose → learnable translator → fixed reconstruct.
    Returns (B, 3, H, W) unbounded residual for additive fusion with main_out.

    Per-channel batching: treats RGB as batch dimension through fixed NSST inverse,
    so same orientation filters reconstruct all 3 channels independently.
    """
    def __init__(self):
        super().__init__()
        self.decomposer    = NSSTDecomposer()
        self.reconstructor = NSSTReconstructor()
        self.proc_low = NSSTScaleProcessor(in_ch=1,  apply_speckle_gate=False)
        self.proc_L1  = NSSTScaleProcessor(in_ch=8,  apply_speckle_gate=True)
        self.proc_L2  = NSSTScaleProcessor(in_ch=8,  apply_speckle_gate=True)
        self.proc_L3  = NSSTScaleProcessor(in_ch=16, apply_speckle_gate=True)
        self.proc_L4  = NSSTScaleProcessor(in_ch=16, apply_speckle_gate=True)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        B, _, H, W = sar.shape
        coefs = self.decomposer(sar)  # [low(B,1,H,W), [D×(B,1,H,W)], ...]

        # Low-pass: (B,1,H,W) → (B,3,H,W) → (B*3,1,H,W)
        low_proc = self.proc_low(coefs[0])
        low_3ch  = low_proc.reshape(B * 3, 1, H, W)

        procs = [self.proc_L1, self.proc_L2, self.proc_L3, self.proc_L4]
        dirs_3ch = []
        for level_idx, proc in enumerate(procs):
            level_coefs = coefs[level_idx + 1]              # list of D tensors (B,1,H,W)
            D = len(level_coefs)
            stacked  = torch.cat(level_coefs, dim=1)        # (B, D, H, W)
            proc_out = proc(stacked)                        # (B, 3*D, H, W)
            proc_3ch = proc_out.reshape(B * 3, D, H, W)
            dirs_3ch.append([proc_3ch[:, k:k+1] for k in range(D)])

        dst_3ch = [low_3ch] + dirs_3ch
        rec = self.reconstructor(dst_3ch)                   # (B*3, 1, H, W)
        return rec.reshape(B, 3, H, W)


if __name__ == '__main__':
    nsst_np = NSST()
    nsst_np.atrous_filters_init()
    nsst_np.shearing_filters_myer()
    
    torch.manual_seed(0)
    np.random.seed(0)
    img_np = np.random.randn(256, 256).astype(np.float32)
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    
    dec = NSSTDecomposer()
    rec = NSSTReconstructor()
    
    coeffs_t = dec(img_t)
    out_t = rec(coeffs_t)
    
    coeffs_np = nsst_np.dec(img_np)
    out_np = nsst_np.rec(coeffs_np)
    
    # Compare outputs
    err = np.abs(out_np - out_t.squeeze().numpy()).max()
    print("Reconstruction max diff with numpy:", err)
