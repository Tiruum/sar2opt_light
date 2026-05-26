# Changelog
Этот файл призван структурировать и упростить понимание коммитов и запусков, кодируя их букво-численной аббревиатурой.

## llwt-v0.4.5 experiments — overfit screen + SubbandDis pathology fix (2026-05-25, later)
Quick <1h overfit-based screen (pure-L1 single-batch capacity, lr 5e-4, 3000 it) + loss-balance probe to pick the 200ep recipe. llwt_v4 untouched.
- **Overfit capacity (health smoke / ceiling):** E0 ConvNeXtV2 **35.38 dB** (smooth); E2 ConvNeXtV2 + novel full-res detail-residual **35.42 dB** (smooth, capacity-neutral — IHaar path already fits L1); E1 SwinV2 **34.71 dB** with instability spikes at it 2000/2500 (transformer needs warmup). => **keep ConvNeXtV2** (Swin lower + unstable; overfit ceiling is iteration-bound and shared, so it confirms health not fine ranking).
- **CRITICAL discovery (fp32-confirmed):** the inherited v4 **SubbandDis** (Haar-coeff PatchGAN) emits logits **~1.8e5** (main-D healthy at [-0.6,1.6]); last FM-feature layer maxabs **~1.2e5**. So gan_sub ~3.5e10, fm_sub ~1e3 — ~10 orders above every perceptual loss. R2 only "trained" because grad-clip(norm=1) renormalised the total → its adversarial signal was ~entirely the (clipped) exploding subband-D, and any added perceptual loss is **completely swamped (zero effect)**. Spectral-norm is present yet does not bound it warm-started (SN-reparam/warm-load interaction; root-cause deferred).
- **Fix for v0.4.5:** disable SubbandDis (`model.dis.subband.enabled=false`, gan_sub/fm_sub=0); drive realism with the healthy **main-D** + a photorealism suite balanced from measured raw terms (warm R2, fp32) so each contributes ~0.3–0.9: gan_main 1.0(0.61), fm_main 10(0.74), msssim 1.0(0.70), lpips 2.0(0.88), ffl 10(0.32), per_band 2.0(0.54), patchnce 0.1(0.49).
- **Added metric:** KID (KernelInceptionDistance, subset_size=50, guarded) alongside FID/LPIPS/SSIM/PSNR.
- **SARAdapter channel-2 A/B (10ep, fair — proj re-init both arms, only channel-2 differs):** the v4 `log|x|` channel is a **folded, redundant double-log** (input is already dB + normalized to [-1,1]; `log(|x|)` conflates darkest/brightest pixels). Replaced with a **log-domain adaptive Lee despeckle** channel (`lee_despeckle` in blocks.py: `out=mean+w·(x−mean)`, `w=clamp(1−σ_n²/var_local)`, σ_n² estimated per-image from 5th-pct local variance — O(N), edge-preserving, zero params). **Despeckle wins photorealism:** FID 201→**180**, KID 0.068→**0.048 (−29%)**, LPIPS 0.365→**0.355**, SSIM 0.305→**0.310**; PSNR ~flat (15.33→15.27). → `sar_channel2: despeckle` is now the default.
- **Novel opt-in:** full-res SAR-conditioned detail-residual head (`model.gen.detail_residual=true`), zero-init (warm-safe); overfit-neutral, may sharpen real-train output. Default OFF.
- **Verification:** E0/E1/E2 overfit PASS; warm R2 ckpt → v45 ConvNeXt G+main-D loads **0 missing / 0 unexpected**; fast_dev_run on the final recipe (subband off, rebalanced) PASS (all 7 criterions; KID guard returns nan on tiny val by design).
- **Recommended 200ep config = `src/models/llwt_v45/config.yaml`** (ConvNeXtV2 + main-D + balanced suite, warm-start R2 ep056, EMA from ep20, linear_decay 50/200). Follow-up arms: detail_residual on; fix+re-enable SubbandDis (proper SN/output-norm, train fresh).

## llwt-v0.4.5 — Balanced fidelity overhaul + Swin ablation + misalignment loss (2026-05-25)
**Goal:** lift the v4 base generator off its PSNR ~14.5 / **SSIM ~0.28** / FID ~81 plateau — raise PSNR **and** SSIM without losing FID — and answer "is SwinV2 better than ConvNeXtV2?". Built as a new self-contained module `src/models/llwt_v45/` (copy of llwt_v4; dotted `v4.5` is not an importable module name) so edits never touch other models.
- **Root cause (verified in code):** the v4 G-objective is ~85% adversarial (`gan_main·1 + fm_main·10 + gan_sub·1 + fm_sub·10`); the only fidelity anchor was per-band wavelet-L1 @2.0 (~0.07 of g_loss). Almost no structural/perceptual signal → SSIM stuck at 0.28. SEN1-2 pairs are imperfectly co-registered (`sar_channels=1` GRD amplitude — no coherence/dual-pol), which caps pixel-aligned SSIM and pushes the GAN to blur.
- **Phase 0 — isolation (copy-then-own):** vendored `SARAdapter/HaarDown/ConvUpsampleBlock` → `blocks.py`; `LLWFormerDiscriminator` → `dis.py`; full sarformer_wb suite + LPIPS/FFL/PatchNCE → `losses.py`; CUT sampler → `patchnce.py`; self-contained `factory.py` (inlined build_*, dropped dead spkdec/pr). Isolation grep clean. gen.py `__main__` smoke (IHaar identity + zero-init) PASS.
- **Phase 1 — loss overhaul (balanced, non-blurring):** enabled **MS-SSIM** (1−MS-SSIM @1.0, direct SSIM lever) + **LPIPS** (AlexNet @1.0) + ported **Focal Frequency Loss** (Jiang ICCV 2021, `FFLLoss` @1.0, global FFT sharpener complementary to local per-band Haar). Adversarial stack + per-band @2.0 unchanged (hold FID).
- **Phase 2 — SAR misalignment loss:** ported `MultiLayerPatchNCE` (CUT, Park 2020) @1.0 — contrastive supervision on the generator's OWN 4-scale encoder features (no external VGG), tolerant to sub-pixel SAR↔optical shift. Added `LLWv4Generator.encode_optical`; query=encoded fake (grad), key=encoded real (no_grad, detached in loss). Sampler MLP params in opt_g.
- **Phase 3 — Swin ablation (one variable):** added `token_output` mode to `HaarStemProjection` returning SwinV2's `(tokens,(H/4,W/4))` contract; backbone-family branch in gen.py installs the right stem. **SwinV2-Tiny = (96,192,384,768) = ConvNeXtV2-Tiny → decoder untouched.** Same Haar front-end in both arms = clean ablation. Switch via 3 config edits (backbone, `weights_ckpt=null`, tb_version).
- **Verification:** fast_dev_run integration smoke (real ConvNeXtV2 + data, all losses) PASS — g_loss=12.26 / d_loss=1.13 finite, criterions=`[gan,fm,msssim,per_band,lpips,ffl,patchnce]`, optimizer integrity OK. SwinV2 forward smoke PASS — out (2,3,256,256) zero-init mid-gray, pyramid `[96@64²,192@32²,384@16²,768@8²]`, encode_optical OK.
- **Ablation arms:** A `llwt-v0.4.5-cnx-overhaul` (ConvNeXtV2 + overhaul + PatchNCE, warm-start from R2 ep56) — the default config; B `llwt-v0.4.5-swin-overhaul` (SwinV2, fresh, +warmup); C headline @9-scene on the winner. Decision gate: SSIM 0.28→≥0.40 with FID ≤~90 by ep30–40.
- **Params:** netG 35.1M + netD 3.6M + criterions 3.1M (PatchNCE MLPs + frozen LPIPS) trainable. PatchNCE adds ~1.5× G-step backbone compute.
- **Version:** v0.4.5 — status: built + smoke-verified, ready to launch Arm A.

## llwt-v0.5.2 — A3 Adversarial Residual Refiner (2026-05-25)
**Goal:** advance the PSNR↔FID Pareto front past the frozen v4-G FID corner. A residual UNet refiner sits on a frozen v4 generator and is trained **adversarially** (not by L1) so it can ADD detail rather than average it away.
- **Why v0.5.1 (pure-L1 refiner) was abandoned:** L1's minimiser is the conditional mean E[opt|sar] — inherently blurry for the ill-posed SAR→opt map. Over 13 ep the L1 refiner slid toward the PSNR corner: **PSNR 15.08→16.08 (+1.0 dB)** but **FID 143→167** and **LPIPS 0.570→0.613** (both worse). `residual_mean` climbed 0.095→0.115 = the refiner learning a low-freq DC/luminance offset, not structure. Frozen-G already sits at the FID corner (ep0 FID=143 = floor), so no L1/SSIM reweighting can advance the front — only a distributional (adversarial) signal can. Kept as ablation row **R0**.
- **A3 design:** pipeline `coarse=G_frozen(sar)` (no_grad) → `residual=Refiner(sar,coarse)` → `refined=(coarse+residual).clamp(-1,1)`. D sees `[sar,refined.detach()]` vs `[sar,opt]`. G(refiner) loss = `gan_main + gan_sub + fm_main(×10) + fm_sub(×10) + l1_weight·L1`. v4 D stack (MainDis 2-scale conditional PatchGAN + SubbandDis Haar) reused; FourierDis off. Manual optimisation, 2 optimisers (opt_d on D, opt_g on refiner — G frozen), batched D real+fake forward, v4 finite-grad + mode-collapse abort gates carried over.
- **`l1_weight=5.0` is the explicit PSNR↔FID dial** (~20% of g_loss): light pixel anchor that bounds adversarial hallucination on the frozen backbone. Decision-gate: if FID stays ≥143 at ep10 with PSNR healthy, drop to 2.0 to loosen the leash.
- **Self-containment (copy-then-import):** `dis.py` (D stack) + `losses.py` (GANLoss, FeatureMatchingLoss) **copied** into `llwt_v5/` (tunable independently of v4 stage). `build_lr_schedulers` + `build_ema_callback` + `LLWv4Generator` imported (stable / frozen, not modified). opt_g/opt_d LRs read from `optimizer.lr_g/lr_d` (2e-4 equal) so the shared scheduler's decay base stays consistent.
- **Params:** refiner 16.3M + D 3.6M trainable (19.9M); frozen G 35.1M.
- **Verification:** `dis.py` Tier-1 smoke PASS; `smoke_a3.py` fast_dev_run (1 train + 1 val batch, real data + real v4 ckpt) PASS — d_loss=1.03, g_loss=2.97 finite, residual_mean=0 (zero-init identity start), G frozen (0 grads), refiner+D train. Ready to launch (`tb_version=llwt-v0.5.2-adv-refine`, max_epochs=60).
- **Version:** v0.5.2

## llwt-v0.1.0 (2026-05-21)
**Architecture:** LLW-Former — wavelet-native SAR→optical generator.
- **Generator (3.6M params):** SARPhysicsFrontEnd (1→3 ch) → Stem Conv (3→96 ch) → LLWTransform (L=3, learnable lifting wavelet, channel-preserving) → PerSubbandEncoder per level (4 WindowSwinBlock stacks: LL/LH/HL/HH, win=8, heads=4, depth=2) → CrossBandAttention at levels 2,3 → PCSG (physics-conditioned subband gating from 7×7 SAR local-variance) → LLWTransform.inverse → 2× ConvNeXtV2-GRN post-decoder → Conv7×7 RGB head + tanh.
- **Discriminator (3.6M params):** main MSPatchGAN (coarse + fine, spectral-norm, IN(SAR) + opt concat) + subband-D (fixed Haar L=1 on optical pair → 12-ch PatchGAN, ndf=32).
- **Novelty (5 axes):** (i) learnable lifting wavelets (Predict/Update CNNs, lazy-init for PR); (ii) per-subband Swin stacks with band-specialised weights; (iii) cross-band attention; (iv) physics-conditioned subband gating; (v) subband-D operating in Haar coefficient space.
- **Physics-informed regulariser:** SpeckleDecoupleLoss anchors level-1 LL to a 5×5 avg-pool despeckled-SAR proxy and pins HH energy to 0.05 — makes lifting interpretable (LL ≈ content, HH ≈ speckle).
- **PR-safeguard regulariser:** PerfectReconLoss tracks `|x - iLLW(LLW(x))|` against fp32/bf16 round-trip drift; weight 0.01.
- **Loss recipe:** LSGAN (main 1.0 + sub 0.5) + FM (main 10.0 + sub 5.0) + L1 50.0 (mandatory pixel anchor per cfrwd-36 / hfgan-12 lesson) + MS-SSIM 1.0 + LAB-chroma 5.0 + WaveletDetailL1 2.0 + AlexNet LPIPS 1.0 + spkdec 0.05 + PR 0.01.
- **Optimiser:** AdamW G (2e-4, wd=0, single param group, no pretrained backbone) + Adam D (1e-4, TTUR). CosineAnnealingWarmRestarts (T_0=20, T_mult=2). EMA(0.999) from ep10. R1 γ=0.5 every 16 steps on main-D only, forced fp32. bf16-mixed forward.
- **Stability:** every wavelet-side residual zero-init (attn out_proj, MLP last linear, CBA proj_out, PCSG last conv, RGB head last conv) → step-0 forward is identity passthrough + zero-init RGB head → mid-grey output. NaN-grad guard on both opt_d and opt_g. Bad-batch skip on non-finite inputs.
- **Test surface:** 61 unit + integration tests, all green. PR round-trip < 1e-10 (fp64), < 1e-5 (fp32). Identity-at-init verified for WindowSwinBlock, CrossBandAttention, full generator.
- **Fresh start.** weights_ckpt=null, resume_ckpt=null. tb_version=`llwt-v0.1.0`.
- **Plan:** ~120 epochs SEN1-2 9-scene subset @ 256×256 bs=6. Mid-train abort gate: ep30 val/PSNR > 14 AND `pu_param_norm > 1e-3` (lifting actually learned). Status: implementation complete, ready to launch.

## hfgan-1 (2026-04-29)
Architecture: ConvNeXtV2-Tiny encoder + bottleneck attention (2-layer Pre-LN, 64 tokens) + U-Net decoder (5 up-blocks, GroupNorm)
Discriminator: Two-scale spectral-norm PatchGAN
Losses: LSGAN + FM (5.0) + optional FFT (1.0) + optional Perceptual (0.1)
Optimizer: AdamW for G (encoder 2e-5 / decoder 2e-4, weight_decay=0.01), Adam for D (2e-4)
Status: ready to train

## hfgan-5 (2026-05-15)
Changes: Added pixel L1 loss (weight 10.0); halved FM weight (10→5); plumbed encoder_lr_scale config knob (0.1). No architectural changes — loss-only fix. Warm-started from hfgan-4 weights_ckpt.
Result: Trained 65 epochs. Semantic hallucinations (fabricated buildings in agricultural fields, invented road segments) persisted. L1 provided pixel anchoring but insufficient without SAR skip connections. Advancing to hfgan-6 with architectural fixes.

## hfgan-6 (2026-05-15)
Changes:
1. SARSkipPyramid — shallow SAR CNN emitting skip features at 128×128 (32ch) and 256×256 (16ch); proj_* convs zero-initialized for safe warm-start. up1/up0 in_channels widened to accept SAR skips.
2. ChannelAdapterV2 — replaced 27-param ChannelAdapter (single Conv+GELU) with 3-conv ImageNet-normalized stem; zero-init to_rgb + bias=ImageNet mean so step-0 backbone input is in-distribution neutral.
3. 2-D sin-cos bottleneck positional encoding — replaced 1-D zero-init learnable Parameter with fixed register_buffer; transformer spatially aware from step 0 with no learning required.
4. Removed dead NSST test file (test_hfgan_nsst.py, 597 lines) and TerraMind dead fixture (MockTerraMindBackbone).
Warm-start: hfgan-5 last.ckpt (checkpoints/huggingface-gan/hfgan-5/last.ckpt), strict=False.
Result: 169 epochs. Best ckpt: PSNR 19.18 dB. Val average: PSNR 18.88 dB, SSIM 0.462, LPIPS 0.462, FID 327. Architecture correct — SAR skips and adapter wired successfully. Failure mode: L1-dominance (weight 10.0) causes chroma collapse to training-set mean (brown soup); fft_weight=0 leaves no high-freq supervision → smudges; perceptual_weight=0.1 too weak for distribution alignment → high FID/LPIPS. Advancing to hfgan-7 with loss rebalance.

## hfgan-7 (2026-05-16)
Changes: Loss recipe rebalance only — no architectural changes. l1_weight 10.0→3.0 (break chroma collapse), fft_weight 0.0→2.0 (high-freq supervision), perceptual_weight 0.1→1.0 (texture/distribution pressure). gan_weight/fm_weight unchanged (1.0/5.0). Short fine-tune: max_epochs=80, linear_decay_epochs=40. Warm-start from hfgan-6 last.ckpt.
Result: 80 epochs. Best ckpt: PSNR 19.24 dB (ep54). Final: PSNR 19.11 dB, SSIM 0.458, AlexNet LPIPS 0.258, VGG LPIPS 0.448. No visual improvement observed. Post-mortem: warm-start from hfgan-6 anchored decoder in the brown-soup local minimum; 80 epochs of rebalanced loss insufficient to escape. Also discovered metric bug — training used AlexNet LPIPS, inference used VGG LPIPS (different scale, not comparable). Advancing to hfgan-8: fresh start + chroma loss.

## hfgan-8 (2026-05-16)
Changes:
1. ChromaLoss — L1 on CIE L*a*b* a*/b* chroma channels only (chroma_weight=5.0); pure-PyTorch rgb_to_lab, float32 cast for bf16 safety; directly attacks mean-color bias without penalizing luminance.
2. Fresh start — no warm-start (weights_ckpt=null); breaks anchor in hfgan-7 brown-soup local minimum.
3. Metric consistency — inference.py switched to AlexNet LPIPS (net_type='alex') matching training metric; best ckpt loaded (ep054) instead of last.ckpt.
4. Extended training — max_epochs=160 (was 80 fine-tune), linear_decay_epochs=40; gives 120 flat-LR epochs for fresh convergence.
Loss: l1=3.0, fft=2.0, perceptual=1.0, chroma=5.0, gan=1.0, fm=5.0.
Result: Collapsed at ep11. val/psnr frozen at 7.43 dB ep11–17, d_loss → 0.007 (D dominated from ep0). Generator locked to constant output (mode collapse). Root cause: chroma_weight=5.0 + no gradient clip → G loss too large → D dominated in early epochs before clip was added. Post-mortem: loss engineering has hit a fundamental limit — decoder has no mechanism to infer which color to produce for ambiguous SAR backscatter (wheat ≈ grass ≈ rapeseed). No loss penalty can fix this. Advancing to hfgan-9 with SPADE architectural conditioning.

## hfgan-9 (2026-05-16)
Changes:
1. SPADENorm — Spatially-Adaptive Denormalization (Park et al. 2019 / GauGAN) replaces GroupNorm in all 5 ConvUpsampleBlock decoder stages. Per-pixel (gamma, beta) predicted by a 2-conv branch from the conditioning map. Zero-init on gamma/beta → identity at step 0, training-stable.
2. Conditioning sources: up4←s2 (384ch), up3←s1 (192ch), up2←s0 (96ch), up1←f128 (32ch SAR skip), up0←f256 (16ch SAR skip). Same skip tensors already flowing via concat now also drive per-pixel normalization.
3. Simplified loss: fft=0.0, perceptual=0.0, chroma=0.0 — architecture handles color, losses handle pixel fidelity only (l1=3.0, fm=5.0, gan=1.0).
4. Fresh start (weights_ckpt=null); gradient clip max_norm=1.0 retained from hfgan-8 patch.
Param cost: ~74.6M base + ~3–5M SPADE (10 SPADENorm modules, hidden=128).
Target: green chroma visible in val/img_001 by ep30; PSNR ≥ 18 dB, AlexNet LPIPS < 0.40 by ep80.
Result: ep83 plateau — PSNR 18.22 dB, AlexNet LPIPS 0.303, FID 342. SPADE gamma/beta convs converged near-zero (identity), conditioning had no measurable effect. Root cause confirmed: local 3×3 SPADE cannot overcome SAR backscatter ambiguity (wheat ≈ grass ≈ rapeseed); no cross-modal reasoning in architecture. Advancing to hfgan-10 with cross-modal attention.

## hfgan-10 (2026-05-16)
Changes:
1. SARBottleneckEncoder — 5-block strided CNN (1→64→128→256→512→768@8²); produces K/V tokens from raw SAR at bottleneck resolution for cross-modal attention.
2. CrossAttentionBlock — Q=decoder features, K/V=encoder/SAR context; to_out zero-init (identity at step 0); 2D sin-cos pos embed on K/V; F.scaled_dot_product_attention (flash attn when available). _pos_cache avoids recomputing 600KB embed each forward pass.
3. BottleneckAttention extended — optical self-attn (unchanged) + cross-attn stream where optical tokens (Q) attend to raw SAR bottleneck (K/V). Sums with residual. Cross-attn to_out zero-init.
4. ConvUpsampleBlock rewritten — SPADE removed. CrossAttentionBlock replaces GroupNorm when cond_ch>0. up4 (cond=s2, 384ch@16²), up3 (cond=s1, 192ch@32²). up2/up1/up0 plain GroupNorm (token count too large).
5. HFGenerator updated — instantiates SARBottleneckEncoder, threads SAR through bottleneck cross-attn, s2/s1 serve dual purpose (skip concat + cross-attn K/V).
6. fm_weight 5.0 → 10.0 (matches cfrwd-36 best run).
7. Fresh start (weights_ckpt=null, resume_ckpt=null).
Param delta: +SARBottleneckEncoder (~3.6M) + CrossAttentionBlock×6 (~4M) vs SPADE removed (~5M) ≈ net +2.6M.
Success criteria: ep10 d_loss >0.05, val/psnr >14 dB; ep30 green chroma in agricultural fields; ep80 LPIPS <0.28, FID <280, PSNR ≥18 dB.
Result: (pending)

## hfgan-11 (2026-05-17)
Hypothesis: discriminator + FM rewarding low-level texture rather than semantic correspondence. Three D-side fixes bundled to free D capacity for structure and stop FM forcing G to copy SAR speckle.
Changes:
1. FinePatchDisBranch — new 3-layer spectral-norm PatchGAN branch (~46×46 RF); replaces the 140×140 second branch. Runs at full 256×256 resolution (no AvgPool downsample). logits shape (B,1,31,31). Returns 3 intermediate features.
2. HFGANDiscriminator.forward — `F.instance_norm` applied to SAR and OPT independently before `cat`. Frees D capacity from absorbing modality brightness shift (empirically ~7×10⁶ attenuation of brightness propagation through D).
3. HFGANDiscriminator.forward — feature lists sliced `feats1[1:] + feats2[1:]` (drop layer 0 of each branch). Layer 0 = post-stride-2 LeakyReLU on raw concat = edges/speckle; matching it in FM is a texture-hallucination pathway (G copies SAR speckle into optical pixels). Combined FM-visible features: 5 (3 from branch1 + 2 from branch2). D logits unchanged.
Loss recipe: unchanged (gan=1.0, fm=10.0, fft=0.0, perceptual=0.1). Optimizer unchanged.
Fresh start: weights_ckpt=null, resume_ckpt=null.
Tests: 48/48 hfgan tests green (incl. 3 new dis tests covering fine-patch shape, per-instance-norm attenuation, layer-0 drop).
Success criteria: ep10 d_loss stable >0.05; ep80 val/lpips improves ≥0.02 vs hfgan-2 baseline (0.303) OR visible reduction in small fake-object hallucinations in val/img_001.
Decision gate: if no metric movement by ep80 → move to Phase B (add L1=100 loss anchor, the more likely root cause per cfrwd-36 audit).
Result: Collapsed within 1 epoch. Same signature as hfgan-8: val/psnr frozen bit-for-bit at 7.42901 dB across all 7 epochs (G mode-collapsed to constant output), train/d_loss → 0.0002 by ep6-7, train/g_loss flat ~1.2. val/lpips 0.914, val/fid 736 (vs hfgan-2 baseline 0.303/280). Root cause: 3 D-side fixes all tilted balance toward D simultaneously, with no pixel-space anchor for G to fall back on. Confirms cfrwd-36 audit: missing L1 is the true root cause; D changes amplified collapse but were not the original sin. Advancing to hfgan-12 with L1=100 + TTUR + wd_g=0 (matching proven cfrwd-36 recipe), keeping all 3 D-side fixes from hfgan-11.

## hfgan-12 (2026-05-17)
Hypothesis: G mode collapse in hfgan-11 is the absence-of-anchor failure mode identified in the cfrwd-36 audit. L1 pixel loss gives G a direct gradient signal independent of D — G can keep learning even when D dominates. TTUR + wd_g=0 align optimizer recipe with cfrwd-36 (the best-image-quality run).
Changes:
1. losses.py — new `L1Loss(nn.Module)` wrapping `F.l1_loss(pred, target)`.
2. factory.py — `build_criterions` instantiates `L1Loss` when `cfg.loss.l1_weight > 0` (gated like fft/perceptual to avoid no-op cost).
3. main.py training_step — adds `criterions['l1'](fake, opt) * loss_cfg.l1_weight` to g_loss (mirrors fft/perceptual gating). Also logs `train/d_real_mean` and `train/d_fake_mean` for early D-dominance detection (abort signal: real-fake > 0.8 at ep1).
4. config.yaml — `loss.l1_weight: 100.0` (NEW; matches pix2pix tradition and cfrwd-36 effective L1≈100 scale).
5. config.yaml — `optimizer.lr_d: 2.0e-4 → 1.0e-4` (canonical TTUR; halves D step size, gives G gradient room).
6. config.yaml — `optimizer.weight_decay_g: 0.01 → 0.0` (cfrwd-36 used wd=0; ConvNeXtV2 has implicit reg via LayerNorm and drop_path; wd=0.01 on pretrained backbone known to drag fine-tuning).
7. config.yaml — `ema.start_epoch: 30 → 10` (hfgan-11 collapsed in 7 ep before EMA even started; lower start preserves option to recover via averaged weights).
D architecture: unchanged (all 3 hfgan-11 D-side fixes preserved — fine-patch branch, instance-norm, FM[1:]).
Loss recipe: gan=1.0, fm=10.0, **l1=100.0 (NEW)**, fft=0.0, perceptual=0.1.
Fresh start: weights_ckpt=null, resume_ckpt=null.
Tests: 55/55 hfgan tests green (3 new: `test_l1_loss_identical_inputs`, `test_l1_loss_known_diff`, `test_l1_loss_grad_flows`; 2 new factory: `test_build_criterions_l1_enabled_when_weight_positive`, `test_build_criterions_no_l1_when_zero`; 2 new main: `test_l1_criterion_present_when_weight_positive`, `test_g_loss_with_l1_anchor_is_finite`).
Success criteria: ep1 train/d_real_mean - train/d_fake_mean < 0.8 (D not dominating); ep10 val/psnr > 12 (escaped 7.43 mode collapse); ep80 val/lpips < 0.40 + visual: fewer fake buildings in val/img_001.
Decision gate: if val/psnr breaks free from 7.43 → continue training to ep160 → compare to hfgan-2 baseline. If collapses again → ablate D fixes one-by-one (instance_norm most suspect).
Result: (pending)

## hfgan-14c (2026-05-18)
Hypothesis: hfgan-14 / hfgan-14b mode-collapsed (val/psnr=7.4290 bit-exact across 83 ep; d_real_mean ≈ d_fake_mean ≈ 0.449 → degenerate Nash). Pure GAN+FM recipe with no pixel anchor cannot recover G from constant tanh-saturated output. Compounding root cause: stale FM real-features in main.py — real_feats captured pre-opt_d.step() reused as FM target post-step → D weights changed → FM target drifts each batch → incoherent gradient.
Changes:
1. main.py training_step — replace `real_feats_d = [f.detach() for f in real_feats]` with a fresh `with torch.no_grad(): _, real_feats_post = self.netD(sar, opt)` after `opt_d.step()`. G's FM target now from same D weights producing fake_feats — coherent gradient.
2. config.yaml — `system.tb_version: "hfgan-14b" → "hfgan-14c"`.
Loss recipe / D architecture unchanged from hfgan-14: gan=1.0, fm=10.0, l1=0.0, fft=0.0, perceptual=0.0, r1_gamma=0.5, r1_every=16. EMA still off.
Tests: API unchanged; pytest tests/test_hfgan_*.py must stay green.
Decision gate (ep10): d_real_mean − d_fake_mean > 0.05 → FM bug confirmed dominant, continue to ep80. Still ≈ 0 → FM fix insufficient, jump to hfgan-15 with l1_weight=100 restored.
Result: (pending)

## Журнал экспериментов (TensorBoard Runs)
Здесь фиксируются цели, параметры и результаты каждого запуска для отслеживания в TensorBoard.

### 21.09.2025
*   **`cfrwd-1`**: *Модель создана, начинаем улучшать, подгонять параметры и прочее, чтобы достичь наилучших результатов.*
    *   **Изменения:** Создание базовой модели CFRWD GAN для последующих экспериментов.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-2`**: *Попытка устранения генерации цветного шума.*
    *   **Изменения:** В дискриминатор добавлена возможность выбора использования conditional (передача SAR) или нет. `main.py` отредактирован под случай 3-канального входного изображения (только fake opt). В конфиг добавлена быстрая настройка этого случая.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-3`**: *Изменение слоев нормализации и свертки в дискриминаторе.*
    *   **Изменения:** В дискриминаторе `nn.InstanceNorm` заменен на `nn.BatchNorm`. Заменен финальный `nn.Conv2d` с параметрами `kernel_size=4, stride=1, padding=1` на `kernel_size=1, stride=1, padding=0`.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-4`**: *Добавление gradient clipping и корректировка функции потерь.*
    *   **Изменения:** Добавлен `torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=0.5)`. Убрано деление в конце `FMLoss`.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-5`**: *Корректировка весов функций потерь и сглаживания меток.*
    *   **Изменения:** Добавлен `L1 loss` с весом * 100, вес `FM loss` повышен до 50. Добавлены параметры `real_label_smooth=0.9` и `fake_label_smooth=0.1`.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-6`**: *Тестирование обучения без аугментаций.*
    *   **Изменения:** Сделана возможность отключать аугментации данных.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-7`**: *Расширение входных данных генератора.*
    *   **Изменения:** Добавлена возможность выбрать три канала на входе генератора вместо одного (SAR теперь может быть не один канал, а три).
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-8`**: *Планирование рефакторинга генератора.*
    *   **Изменения:** Поставлена задача переработать структуру генератора, взяв нормальные ResBlock'и.
    *   **Результат:** N/D
    *   **Версия:** N/D

##### Commit: CFRWD-2
*   **`cfrwd-9`**: *Исправление шумов, вызванных повторной инициализацией.*
    *   **Изменения:** Вынес инициализацию модулей (HFCF) в конструктор генератора, чтобы они не создавались заново каждый раз.
    *   **Результат:** N/D
    *   **Версия:** N/D
    
*   **`cfrwd-10`**: *Исправление ошибочной инициализации фиксированных весов.*
    *   **Изменения:** Исправлена ошибка, при которой `_initialize_weights` инициализировал веса `Haar`-преобразования, которое должно быть фиксированным.
    *   **Результат:** Предположительно, должно убрать артефакты в виде пятен, так как Haar перестал вырождаться в обычный downsampling.
    *   **Версия:** N/D

*   **`cfrwd-11`**: *Рефакторинг кода и обновление датасета.*
    *   **Изменения:** Поправлен `.gitignore` (добавлены файлы `src/data`). Внесены косметические улучшения в `src/data/sen12/dataset.py`. Загружен новый полный датасет SEN1-2 (43 ГБ).
    *   **Результат:** N/D
    *   **Версия:** N/D

##### Commit: CFRWD-9
*   **`cfrwd-12` / `cfrwd-17`**: *Полная переработка ядра модели для стабилизации генерации.*
    *   **Изменения:** Полностью переписаны ветви CFR и HFCF, а также структурные блоки.
    *   **Результат:** Достигнута генерация изображений, визуально похожих на правду. Цели ветки `cfrwd-12` выполнены.
    *   **Версия:** N/D

##### Commit: CFRWD-10
*   **`cfrwd-18`**: *Эксперимент с BatchNorm в декодере и переименование слоя.*
    *   **Изменения:** В `DecoderBlock` заменен `nn.InstanceNorm2d(out_channels, affine=True)` на `nn.BatchNorm2d(out_channels, affine=True)`. Переименован слой `fusion_coef` в `fusion_conv`.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-19`**: *Откат к InstanceNorm в декодере.*
    *   **Изменения:** В `DecoderBlock` возвращен `nn.InstanceNorm2d(out_channels, affine=True)` (откат `cfrwd-18`). Сохранено имя `fusion_conv`.
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-20`**: *Откат имени слоя fusion.*
    *   **Изменения:** Имя слоя `fusion_conv` возвращено к исходному `fusion_coef` (откат переименования из `cfrwd-18`).
    *   **Результат:** N/D
    *   **Версия:** N/D

*   **`cfrwd-21`**: *Массовый переход на BatchNorm.*
    *   **Изменения:** Произведена повсеместная замена `InstanceNorm` на `BatchNorm` в ключевых компонентах модели.
    *   **Результат:** N/D
    *   **Версия:** N/D

### 14.12.2025
##### Commit: 
*   **`cfrwd-22`**: *Обновить дискриминатор для лучшей генерации.*
    *   **Изменения:** Изменил структуру дискриминатора, теперь forward прогоняется дважды. Редактировал `main.py`, добавил отправку изображений в телеграм. Структурировал вывод логов в tensorboard. Сделал общую функцию `cleanup_memory`.
    *   **Результат:** Визуализации сильно улучшились, пропали артефакты, нормализовались цвета, `PSNR = 14.5414` (максимально полученный cfrwd-20: 14.1507) `SSIM = 0.2438` (максимально полученный cfrwd-20: 0.1581).
    *   **Версия:** v2.2.0

*   **`cfrwd-23`**: *Запускаю обучение на всем датасете sen1-2 (2.54 Гб) на 200 эпох*
    *   **Изменения:** Нет.
    *   **Результат:** Лучшие метрики: `PSNR = 16.75`, `SSIM = 0.2945`. Результаты были хорошими в начале, под конец стали не очень (начался сильный разброс в `val (l1, psnr, ssim)`). `train/loss_gan` стал равен 1, а train/loss_d стал равен нулю. Генерация, соответственно, тоже испортилась. Это значит, что дискриминатор начал доминировать над генератором.
    *   **Версия:** v2.2.0

*   **`cfrwd-24`**: *Запускаю обучение на 0.3 датасете sen1-2 (2.54 Гб) на 150 эпох*
    *   **Изменения:** Добавил TTUR: `lr_g: 2e-4`, `lr_d: 1e-4`.
    *   **Результат:** 
    *   **Версия:** v2.2.1

### 15.02.2026
*   **`cfrwd-25`**: *Исправления в main.py*
    *   **Изменения:** Нормализация `loss_gan` по числу скейлов. Оптимизация вычислений через `torch.no_grad()`, фикс записи чекпоинт-файлов (теперь корректно логируется `loss_l1`), добавил scheduler step в `on_train_epoch_end`. Сделал лог `lr_G` и `lr_D` для проверки shedulers.
    *   **Результат:** Изменения в генерации и графиках несущественны
    *   **Версия:** v2.2.2
    
### 19.02.2025
*   **`cfrwd-26`**: *Исправления и оптимизация*
    *   **Изменения:** Исправлено
        Данные:
        1. КРИТИЧНО: аугментации при обучении были отключены
        datamodule.py — train_common_transform был закомментирован (None). Модель обучалась без геометрических аугментаций вообще. Теперь добавлен параметр use_augmentation, управляемый через конфиг use_train_common_transform: true.

        2. КРИТИЧНО: отсутствовал linear_decay_epochs в конфиге
        config.yaml:33 — factory.py ссылается на cfg.scheduler.linear_decay_epochs, но этого ключа не было. Добавлено linear_decay_epochs: 100.

        3. Добавлен RandomRotate90 в аугментации
        transforms.py:14 — спутниковые снимки инвариантны к повороту на 90°. Это практически бесплатная аугментация, которая кратно увеличивает разнообразие данных. Также переупорядочены трансформы (дешёвые flips перед дорогим Affine) и снижена p Affine с 0.9 до 0.7.

        4. Добавлен drop_last=True в train_dataloader
        datamodule.py:84 — без этого последний неполный батч мог вызывать нестабильность BatchNorm при малых размерах.

        5. Защита persistent_workers при num_workers=0
        datamodule.py:36 — persistent_workers=True с num_workers=0 вызывает ошибку PyTorch. Теперь автоматически выставляется False и prefetch_factor=None.

        6. Упрощено чтение SAR в dataset.py
        dataset.py — вместо IMREAD_UNCHANGED + разветвлённая конвертация BGRA/BGR/Gray, теперь:

        sar_channels=1 → IMREAD_GRAYSCALE (всегда 2D, без лишних конвертаций)
        sar_channels=3 → IMREAD_COLOR + BGR→RGB
        Также добавлен __repr__ для удобной отладки.

        7. Проброс use_augmentation в train.py и main.py
        train.py и main.py обновлены для передачи флага из конфига.

        Оптимизировано
        - dataset.py — _ALLOWED_EXT вынесен в frozenset на уровне класса; кэширование os.path.isfile/os.path.join в локальные переменные в _collect_items
        - datamodule.py — make_dataset теперь передаёт classes и items напрямую через конструктор вместо пересканирования файловой системы (было 3 полных прохода по FS → теперь 1)

        1. Генератор:
        - gen.py — CFRBlock.forward: промежуточные self.down(p1), self.down(q1), self.up(q3), self.up(k4) кэшируются в переменные вместо повторного вычисления (экономия ~12 операций down/up на каждый forward pass)

        2. Тренировка:
        - main.py — убран лишний self.netG.to(device) (Lightning делает это сам); визуализация обёрнута в torch.no_grad(); модуль-level импорты перемещены внутрь __main__ (не выполняются при from ... import)
        - train.py — убрана двойная загрузка конфига
        - factory.py — ленивая загрузка конфига через @lru_cache; дефолты аргументов не привязаны к module-level переменным
        - losses.py — убран неиспользуемый импорт OmegaConf/config

        3. Utils:
        - cleanup_memory.py — убраны нерабочие global model, dm (эти глобалы существовали только в train.py, а не в модуле cleanup_memory)
        - notification.py — бот создаётся один раз и переиспользуется (singleton _get_bot())
    *   **Результат:** N/A
    *   **Версия:** v2.2.3

*   **`cfrwd-27`**: *Оптимизация генератора, доработка утилит*
    *   **Изменения:** Сигмоида в коэфициенте смешения, новая нормализация генератора, доработка функции очистки (теперь освобождается память после персистент воркерс и др)
    *   **Результат:** На 10 эпохе полного датасета результаты вполне приемлемые для 10 эпохи.
    *   **Версия:** v2.2.4

*   **`cfrwd-28`**: *Смешанная точность для ускорения и возобновление с чекпоинта*
    *   **Изменения:** Добавил смешанную точность, каст для метрик, возобновление с чекпоинта
    *   **Результат:** Время выполнения одной эпохи сократилось с 6:30 минут до 4:30.
    *   **Версия:** v2.3.5

### 21.02.2026
*   **`cfrwd-29`**: *BS=4, везде InsanceNorm*
    *   **Изменения:** BatchSize=4, везде InsanceNorm, max_epoch=800, linear_decay с 400 эпохи. Fusion по формуле `cfr_out + self.fusion_weight * hfcf_out`
    *   **Результат:** 
    *   **Версия:** v2.4.5

*   **`cfrwd-29`**: *Unconditional Dis -> Conditional Dis*
    *   **Изменения:** Unconditional Dis -> Conditional Dis
    *   **Результат:** 
    *   **Версия:** v2.5.5

*   **`cfrwd-31`**: *Добавил EMA, добавил выходы cfr и hfcf веток на картинки, instance_norm -> spectral_norm только в дискриминаторе, перешли на pytorch lightning 2.5.5*
    *   **Изменения:** Добавил EMA, добавил выходы cfr и hfcf веток на картинки, instance_norm -> spectral_norm только в дискриминаторе
    *   **Результат:** (Какие метрики/визуализация получились. Важно!)
    *   **Версия:** v2.6.5

*   **`cfrwd-32`**: *Архитекктура CFR была переписана*
    *   **Изменения:**
    1. Исправлено "бутылочное горлышко" разрешения (Final Fusion)
    Было: В конце CFRBlock ветка максимального разрешения k1 (256x256) сжималась до 128x128, а потом в декодере делался Upsample. Это приводило к безвозвратной потере мелких деталей.
    Стало: Теперь слияние (fuse3_to4) происходит на максимальном разрешении. Ветки k2, k3, k4 апсэмплятся до размера k1 (256x256) и конкатенируются. Из-за этого из CFRBranch.decoder убран лишний Upsample — сеть больше не теряет пространственную информацию.
    2. Решена проблема "голодания" каналов (Channel Starvation)
    Было: Каналы падали с 64 до 4. Свертка на 4 каналах не могла выучить ничего сложного. При этом в начале делались тяжелейшие вычисления на 64 каналах при разрешении 256x256.
    Стало: Я внедрил логику HRNet. Теперь каналы распределяются пропорционально разрешению: c1=16, c2=32, c3=64, c4=128.
    Перед входом в ResBlock первого этапа добавлены проекции proj11 и proj12 (1x1 свертки), которые сразу понижают каналы до нужного уровня.
    Это дает сети огромную емкость на глубоких слоях (где разрешение маленькое, а каналов много — 128) и сильно экономит видеопамять на начальных слоях.
    3. Добавлена нелинейность при слиянии (Non-linear Fusion)
    Было: Слои fuse состояли только из ReflectionPad2d и Conv2d. Это линейное преобразование.
    Стало: Все слои fuse (от fuse1_to2_1 до fuse3_to4) теперь обернуты в nn.Sequential с InstanceNorm2d и LeakyReLU(0.2). Теперь сеть может строить сложные нелинейные комбинации признаков с разных масштабов.
    *   **Результат:** Качество генерации на train кратно возрасло. Однако fusion зануляет HFCF ветку. Дополнительно, модель галлюционирует из-за отсутствия L1 лосс, но мы пока не торопимся его добавлять. Будем пробовать фьюзить не две картинки, а CFR и фичи.
    *   **Версия:** v2.7.5



### 03.04.2026
*   **`cfrwd-33`**: *Полная переработка HFCF ветки + AdaptiveFusion (logit-level)*
    *   **Изменения:**
    1. Исправлена "атрофия" HFCF ветки (fusion_weight → 0.2)
    Причина: ранний суммарный вход `hfcf_g2 = hfcf_g2_in + hfcf_g3` смешивал две группы вейвлет-коэффициентов до потоков, не давая ветке специализироваться. Плюс скалярный `fusion_weight` позволял градиенту глобально гасить ветку.
    Стало: три независимых потока — (a) LL2 @ W/4 (низкочастотный), (b) [LH2,HL2,HH2] @ W/4 (средние частоты), (c) [LH1,HL1,HH1] @ W/2 → W/4 (высокие частоты). Каждый поток обрабатывается независимо через HFCFPreprocess → CBAM → ResBlocks, затем объединяются через 1×1 Conv.
    2. Добавлен CBAM (Channel + Spatial Attention) перед каждым потоком
    Назначение: фильтрация спекл-шума SAR в вейвлет-коэффициентах. Channel attention отбирает информативные каналы, Spatial attention маскирует шумные регионы. Reduction ratio = 4, минимум 4 канала.
    3. Заменён скалярный `fusion_weight` на `AdaptiveFusion` (пространственная карта весов, слияние на уровне логитов 3ch)
    Было: `fused = cfr_out + fusion_weight * hfcf_out` (один скаляр на весь батч).
    Стало: `AdaptiveFusion` конкатенирует cfr и hfcf логиты → Conv → softmax по dim=1, возвращает взвешенную сумму с per-pixel весами. Ветки конкурируют за каждый пиксель независимо.
    *   **Результат:** HFCF деградирует к ~нулевому плоскому выходу к эпохе 30. AdaptiveFusion слишком быстро учит w_cfr≈1 т.к. 6ch входа недостаточно. Вторичная атрофия.
    *   **Версия:** v2.8.5

*   **`cfrwd-34`**: *Feature-level fusion + FFT Loss + AdaptiveLoss (Kendall et al.)*
    *   **Изменения:**
    1. Перенос AdaptiveFusion с уровня логитов (3ch) на уровень feature maps (32ch)
    Было: каждая ветка сама доходила до 3ch через свой FinalDecoderBlock → AdaptiveFusion(6ch) → tanh.
    Стало: ветки отдают 32ch feature maps → AdaptiveFusion(64ch) → единый FinalDecoderBlock (32→3) → tanh.
    Смысл: градиент от лосса проходит в обе ветки через общий финальный слой независимо от весов fusion — HFCF физически не может получить нулевой градиент.
    2. AdaptiveFusion возвращает (fused_feats, weights): веса логируются как fusion/w_hfcf и fusion/spatial_std.
    3. Добавлен FFTLoss: L1 на лог-магнитуде 2D FFT спектра.
    log(1+|F|) сжимает динамический диапазон → сбалансированный градиент на всех частотах.
    float32 computation: FFT нестабилен в bf16.
    4. Добавлен AdaptiveLoss (Kendall et al., 2018): авто-балансировка L1 и FFT через обучаемые eta.
    Формула: L = sum_i(L_i * exp(-eta_i) + eta_i). eta обучаются вместе с G (включены в optG).
    Убран ручной l1_weight из конфига — веса определяются автоматически.
    5. LPIPS как val-метрика (всегда) и опциональный training loss (use_lpips: false/true).
       - val/lpips — всегда логируется, независимо от use_lpips.
       - При use_lpips=true: LPIPS добавляется в AdaptiveLoss третьим компонентом [L1, FFT, LPIPS].
       - LPIPSLoss использует замороженный AlexNet (не обновляется, только градиент в fake_opt).
    6. Новые метрики в TensorBoard:
       - fusion/w_hfcf, fusion/spatial_std — здоровье HFCF ветки
       - loss/eta_l1, loss/eta_fft, (loss/eta_lpips) — текущие логи дисперсий
       - loss/w_l1, loss/w_fft, (loss/w_lpips) — exp(-eta): эффективные веса
    *   **Результат:** (Какие метрики/визуализация получились. Важно!)
    *   **Версия:** v2.9.5

### 06.04.2026
*   **`cfrwd-35`**: *Углубленный анализ и подтверждение стабильности на BS=8, LR-scaled.*
    *   **Изменения:**
    1. Увеличение batch_size: 4 → 8. LR масштабирован по √8: `lr_g/lr_d: 2e-4 × √8 = 5.6e-4`.
    2. Снижение max_epochs с 1600 → 800 (эквивалент 200 эпох × BS=8/BS=1 с осознанным урезанием для быстрого тестирования).
    3. Исправление багов накопления метрик: замена прямого вызова `metric(pred, target)` на правильный паттерн `metric.update()` + `metric.reset()` — предотвращена утечка GPU памяти (SpectralAngleMapper хранил полные тензоры).
    4. Отключение cuDNN benchmark (`benchmark: false`) — препятствовано выделению 16 ГБ workspace в CFRBlock/HFCFBranch с 50+ уникальными формами.
    5. Снижение num_workers: 8 → 3 — уменьшена фрагментация памяти при spawn на Windows (6+6=12 процессов → 3+3=6 процессов, экономия ~2.6 ГБ).
    *   **Результат:** Стабильное обучение на 89+ эпохах. Окончательные метрики: val/psnr = 15.17 дБ, val/ssim = 0.185, val/ergas = 605 (некорректен из-за ошибки range). Модель не показала признаков divergence, VRAM стабилен на ~12 ГБ.
    *   **Версия:** v2.9.5

*   **`cfrwd-36`**: *Критический аудит: исправление 7 глубоких багов метрик, градиентов, памяти и инициализации.*
    *   **Изменения:**
    1. **BUG-1 (Метрики):** ERGAS и SAM теперь вычисляются на `[0,1]` изображениях вместо `[-1,1]` → исправлено ERGAS=600+ и SAM=NaN. Добавлены guard-ы `isfinite()` / `isnan()` в metric_handler для диагностики.
    2. **BUG-2 (HaarDown):** Упрощена инициализация — масштаб применён один раз при регистрации буфера (`weight *= 0.5`), убран избыточный `* self.scale` в forward. Удалено неиспользуемое `import math`. Добавлено `.clone()` при регистрации буфера для предотвращения класс-уровневой мутации.
    3. **BUG-3 (EMAWeightAveraging):** Исправлено начало EMA с epoch=0 (неправильно) на epoch=30 (`update_starting_at_epoch`). Переопределён `on_train_batch_end` для передачи `epoch_idx` в обновление. Добавлена валидация `update_every_n_steps < 1`.
    4. **BUG-4 (LPIPSLoss VRAM):** LPIPSLoss теперь разделяет AlexNet backbone с val-метрикой вместо загрузки дублирующейся копии — экономия 60 МБ VRAM. `build_criterions` требует явно передавать `lpips_backbone` при `use_lpips=true`.
    5. **BUG-5 (build_optimizers):** Замена `or` на `if x is None:` guards — исправлены молчаливые переопределения `lr=0` или `beta1=0`.
    6. **BUG-6 (VRAM кэш после визуализации):** Добавлен `torch.cuda.empty_cache()` в `on_train_epoch_end` после сохранения визуализации.
    7. **BUG-7 (CFRBlock redundant ops):** Кэширование `q1_dd = self.down(q1_d)` и `q2_dd = self.down(q2_d)` в CFRBlock — устранены redundant AvgPool2d вызовы в cross-fusion.2.
    *   **Результат:** ERGAS теперь корректен (~150–200 на валидации). SAM без NaN. VRAM стабилен (~12 ГБ, без скачков после каждой эпохи). EMA начинает обновление с эпохи 30, а не шага 0. LPIPSLoss требует явного backbone при инициализации (breaking change). Тренировка более предсказуема и диагностична.
    *   **Версия:** v3.0.0

## sarformer-wb-1 (2026-05-20)
Fresh-start hybrid SAR→optical generator, new package `src/models/sarformer_wb/`, replacing the plateaued `huggingface_gan/hfgan-{6..18}` lineage.
Architecture:
1. Encoder — Swin V2-Tiny (`microsoft/swinv2-tiny-patch4-window8-256`, ImageNet-22k pretrained, `out_indices=[1,2,3,4]`, lr×0.1).
2. SARPhysicsFrontEnd — [raw, log1p, reflect-padded Sobel, predicted speckle log-variance `s_spk`] → 4ch input concatenation; learned `s_spk` head feeds both the wavelet bottleneck and the Phi-physics-D.
3. WaveletBottleneck — Haar DWT level-1 on the (768, 8, 8) Swin V2 final feature; LL refined by a MiniSwinV2Block (scaled-cosine attn + log-CPB); LH/HL/HH refined by a SpeckleGatedConvStack (1×1 project → 3 reflect-pad convs → 1×1 expand, gated by sigmoid(s_spk downsampled)); iDWT recombines with a learnable residual gate (init sigmoid(-4) ≈ identity).
4. Decoder — 5 stages of `DecoderStage` = PixelShuffle (ICNR init) + ConvNeXtV2 GRN block (DWConv7×7 + LN + Linear + GELU + GRN + Linear + DropPath); GRN gamma/beta zero-initialised so the layer starts as identity.
5. SAR cross-attention skips — lightweight windowed cross-attention (`SARCrossAttnSkip`, single-head, inner_dim=64, window=8) at decoder resolutions 32 and 64 with Q = decoder feature, K/V = SAR pyramid feature (3 stride-2 conv stages on raw SAR); learnable scalar gate inits ≈ 0 (tanh) so the skip starts as identity.
6. Heads — RGB head = ReflectionPad+Conv7×7 + tanh; uncertainty head = Conv3×3 producing per-pixel log-variance `s(x)` (zero-initialised, training-only).

Discriminators:
- `MSPatchGANDis` — 2-scale conditional PatchGAN (70×70 coarse + 46×46 fine), spectral-norm, asymmetric input normalisation (SAR instance-normed, optical raw), layer-0 features dropped from FM (`hfgan-18` lesson). 22×22 micro branch from hfgan-18 dropped — micro-texture coverage moves to the Phi-D side.
- `PhiPhysicsDis` (Φ-GAN-inspired) — 4-layer spectral-norm PatchGAN on `[SAR, optical, s_spk]` (5ch). Frozen for `phi.freeze_epochs=5` then trained alongside the main D.

Losses (config weights):
- LSGAN main (1.0) + FM main (10.0)
- LSGAN phi (0.5) + FM phi (5.0)
- `UncertaintyL1Loss` (10.0) — Kendall-Gal aleatoric weighting `Σ exp(-s)·|y-ŷ|₁ + s` + `0.01·|s|` reg, log-var clamped to [-2,+2]; replaces plain L1.
- MS-SSIM (1.0), LAB-chroma (5.0), Wavelet-detail L1 on Haar LH/HL/HH (1.0)
- `SpeckleConsistencyLoss` (0.5, linear warm-up epochs 5→10) — reflectivity proxy head + Gamma(L=4, 1/L) sample modulated by s_spk; reflectivity head's params live in the G optimiser.
- R1: main-D γ=0.5 every 16; Phi-D γ=0.5 every 32.

Training recipe: AdamW G (encoder lr 2e-5, fresh 2e-4) + Adam D (1e-4 each), β=(0.5,0.999), bf16-mixed, BS=8, 80 epochs (cosine warm restarts with 60-epoch linear decay tail), EMA decay=0.999 from ep 10, gradient clip G=1.0 / D=5.0, torch.compile on G + main-D (Phi-D off — gamma-sampling not compile-friendly).

Param budget (real Swin V2): G≈50.6 M, D_main≈3.4 M, D_phi≈0.7 M → grand total ≈54.7 M trainable + 26.3 M frozen (FID Inception + LPIPS AlexNet). Under the 100 M ceiling.

Verification:
- Pytest: 60/60 sarformer_wb tests pass (gen/dis/losses/factory/main); 112/112 hfgan-18 tests still pass (no regression).
- Smoke train (1 epoch, BS=2, 4 train + 4 val batches) on RTX 4080 Super 16 GB completes without NaN/OOM. Both d_loss and d_phi_loss compute (Phi-D path exercised with `freeze_epochs=0`). Speckle-consistency loss fires (`warmup_start=0`). Validation metrics finite (PSNR 5.24 dB, SSIM 0.029, LPIPS 0.75, FID 652 — expected for an initialised model).

Status: ready to launch ablation A1 (no Phi-D, no speckle-consistency) and ablation A2 (full configuration).

### DD.MM.YYYY
*   **`cfrwd-XX`**: *Краткая цель эксперимента.*
    *   **Изменения:** (Что именно меняли в коде/параметрах)
    *   **Результат:** (Какие метрики/визуализация получились. Важно!)
    *   **Версия:** vX.Y.Z (X — мажорная версия, Y — минорная версия и Z — патч-версия)