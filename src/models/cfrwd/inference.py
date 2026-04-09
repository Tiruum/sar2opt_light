import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torchmetrics.image import (
	ErrorRelativeGlobalDimensionlessSynthesis,
	LearnedPerceptualImagePatchSimilarity,
	PeakSignalNoiseRatio,
	SpectralAngleMapper,
	StructuralSimilarityIndexMeasure,
)

from src.data.sen12.datamodule import SEN12Datamodule
from src.models.cfrwd.gen import CFRWDGenerator
from src.utils.visualize import visualize_batch


DEFAULT_CONFIG_PATH = Path("src/models/cfrwd/config.yaml")
DEFAULT_CKPT_PATH = Path("checkpoints/cfrwd/cfrwd-36/epoch=264-psnr=15.4282.ckpt")
DEFAULT_OUTPUT_PATH = Path("output/cfrwd/images/inference/val_preview_cfrwd-36_epoch264.png")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Show SAR | Generated | Ground Truth triplets from SEN12 validation set."
	)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
	parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT_PATH, help="Path to CFRWD checkpoint")
	parser.add_argument("--batch-idx", type=int, default=0, help="Validation batch index to preview")
	parser.add_argument("--max-rows", type=int, default=6, help="How many samples to show from the batch")
	parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save preview image")
	parser.add_argument("--show", dest="show", action="store_true", help="Open preview window after save")
	parser.add_argument("--no-show", dest="show", action="store_false", help="Do not open preview window")
	parser.set_defaults(show=True)
	return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
	requested = str(requested_device).lower()
	if requested.startswith("cuda"):
		if torch.cuda.is_available():
			return torch.device("cuda")
		print("CUDA requested in config, but CUDA is not available. Falling back to CPU.")
	return torch.device("cpu")


def load_generator_from_checkpoint(ckpt_path: Path, in_channels: int, device: torch.device) -> CFRWDGenerator:
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

	try:
		checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
	except TypeError:
		checkpoint = torch.load(str(ckpt_path), map_location=device)
	state_dict = checkpoint.get("state_dict", checkpoint)
	if not isinstance(state_dict, dict):
		raise TypeError("Unsupported checkpoint format: expected dict with model weights")

	gen_state = {
		key.removeprefix("netG."): value
		for key, value in state_dict.items()
		if key.startswith("netG.")
	}
	if not gen_state:
		raise KeyError("Generator weights with prefix 'netG.' were not found in checkpoint")

	model = CFRWDGenerator(in_channels=in_channels).to(device)
	load_result = model.load_state_dict(gen_state, strict=False)
	if load_result.missing_keys or load_result.unexpected_keys:
		raise RuntimeError(
			"Checkpoint is incompatible with current generator. "
			f"Missing: {load_result.missing_keys}; Unexpected: {load_result.unexpected_keys}"
		)

	model.eval()
	return model


def get_val_batch(val_loader, batch_idx: int):
	if batch_idx < 0:
		raise ValueError("batch_idx must be >= 0")

	for idx, batch in enumerate(val_loader):
		if idx == batch_idx:
			return batch
	raise IndexError(f"Validation batch index {batch_idx} is out of range")


def show_saved_image(image_path: Path, batch_idx: int) -> None:
	image = plt.imread(str(image_path))
	plt.figure(figsize=(14, 7))
	plt.imshow(image)
	plt.title(f"Validation batch {batch_idx}: SAR | Generated | Ground Truth")
	plt.axis("off")
	plt.tight_layout()
	plt.show()


def compute_preview_metrics(fake_optical: torch.Tensor, real_optical: torch.Tensor, max_rows: int):
	n_samples = min(int(max_rows), int(fake_optical.size(0)))
	psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(fake_optical.device)
	ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(fake_optical.device)
	lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(fake_optical.device)
	sam_metric = SpectralAngleMapper().to(fake_optical.device)
	ergas_metric = ErrorRelativeGlobalDimensionlessSynthesis(ratio=1).to(fake_optical.device)

	rows = []
	for i in range(n_samples):
		pred = fake_optical[i:i + 1].float()
		target = real_optical[i:i + 1].float()
		pred_01 = (pred + 1.0) * 0.5
		target_01 = (target + 1.0) * 0.5

		psnr_metric.reset()
		psnr_metric.update(pred, target)
		psnr_value = float(psnr_metric.compute().detach().cpu().item())

		ssim_metric.reset()
		ssim_metric.update(pred, target)
		ssim_value = float(ssim_metric.compute().detach().cpu().item())

		lpips_metric.reset()
		lpips_metric.update(pred, target)
		lpips_value = float(lpips_metric.compute().detach().cpu().item())

		sam_metric.reset()
		sam_metric.update(pred_01, target_01)
		sam_value_tensor = sam_metric.compute().detach().cpu()
		sam_value = float(sam_value_tensor.item()) if not torch.isnan(sam_value_tensor) else float("nan")

		ergas_metric.reset()
		ergas_metric.update(pred_01, target_01)
		ergas_value_tensor = ergas_metric.compute().detach().cpu()
		ergas_value = float(ergas_value_tensor.item()) if torch.isfinite(ergas_value_tensor) else float("nan")

		rows.append({
			"idx": i,
			"psnr": psnr_value,
			"ssim": ssim_value,
			"lpips": lpips_value,
			"sam": sam_value,
			"ergas": ergas_value,
		})

	return rows


def _mean_ignore_nan(values) -> float:
	tensor = torch.tensor(values, dtype=torch.float32)
	if tensor.numel() == 0:
		return float("nan")
	return float(torch.nanmean(tensor).item())


def print_preview_metrics(rows) -> None:
	if not rows:
		print("No metrics to report.")
		return

	print("\nPer-image metrics (shown samples):")
	print("idx\tpsnr\tssim\tlpips\tsam\tergas")
	for row in rows:
		print(
			f"{row['idx']:02d}\t"
			f"{row['psnr']:.4f}\t"
			f"{row['ssim']:.4f}\t"
			f"{row['lpips']:.4f}\t"
			f"{row['sam']:.4f}\t"
			f"{row['ergas']:.4f}"
		)

	print("Average\t"
		  f"{_mean_ignore_nan([r['psnr'] for r in rows]):.4f}\t"
		  f"{_mean_ignore_nan([r['ssim'] for r in rows]):.4f}\t"
		  f"{_mean_ignore_nan([r['lpips'] for r in rows]):.4f}\t"
		  f"{_mean_ignore_nan([r['sam'] for r in rows]):.4f}\t"
		  f"{_mean_ignore_nan([r['ergas'] for r in rows]):.4f}")


def main() -> None:
	args = parse_args()
	if args.max_rows < 1:
		raise ValueError("max_rows must be >= 1")

	if not args.config.exists():
		raise FileNotFoundError(f"Config not found: {args.config}")

	cfg = OmegaConf.load(str(args.config))
	device = resolve_device(cfg.system.device)

	preview_batch_size = max(int(cfg.data.batch_size), int(args.max_rows))
	dm = SEN12Datamodule(
		data_dir=cfg.data.data_dir.sen12,
		batch_size=preview_batch_size,
		image_size=cfg.data.image_size,
		num_workers=0,
		persistent_workers=False,
		prefetch_factor=2,
		train_val_split_ratio=cfg.data.train_val_split_ratio,
		seed=cfg.data.seed,
		sar_channels=cfg.data.sar_channels,
		use_augmentation=False,
	)
	dm.setup(stage="fit")

	model = load_generator_from_checkpoint(
		ckpt_path=args.ckpt,
		in_channels=int(cfg.model.gen.in_channels),
		device=device,
	)

	real_sar, real_optical = get_val_batch(dm.val_dataloader(), args.batch_idx)
	real_sar = real_sar.to(device)
	real_optical = real_optical.to(device)

	with torch.inference_mode():
		fake_optical, _ = model(real_sar)

	metrics_rows = compute_preview_metrics(
		fake_optical=fake_optical,
		real_optical=real_optical,
		max_rows=args.max_rows,
	)

	args.out.parent.mkdir(parents=True, exist_ok=True)
	visualize_batch(
		real_sar=real_sar.cpu(),
		fake_optical=fake_optical.cpu(),
		real_optical=real_optical.cpu(),
		save_path=str(args.out),
		max_rows=args.max_rows,
		mode="fast",
	)

	print(f"Saved validation preview to: {args.out}")
	print_preview_metrics(metrics_rows)
	if args.show:
		show_saved_image(args.out, args.batch_idx)


if __name__ == "__main__":
	main()
