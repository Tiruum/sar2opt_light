import os
from dotenv import load_dotenv
import telebot
import torch

load_dotenv()

TELEGRAM_BOT_TOKEN= os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_RECIEVER_USER_ID = os.getenv('TELEGRAM_RECIEVER_USER_ID')

def send_telegram(message="", image_path=None):
    """
    Универсальная функция для отправки сообщений и изображений через telebot.

    Args:
        message (str): Текст сообщения. Можно отправлять отдельно или как подпись к фото.
        image_path (str, optional): Путь к файлу изображения на диске.
    """
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

        # Если передан путь к изображению
        if image_path:
            # Проверяем существование файла
            if not os.path.exists(image_path):
                print(f"❌ [TG BOT] Файл не найден: {image_path}")
                return

            # Открываем файл и отправляем как документ
            with open(image_path, 'rb') as doc:
                bot.send_document(TELEGRAM_RECIEVER_USER_ID, doc, caption=message)

        # Если передан только текст (без изображения)
        elif message:
            # Отправляем текстовое сообщение
            bot.send_message(TELEGRAM_RECIEVER_USER_ID, message)

    except Exception as e:
        print(f"❌ [TG BOT] Произошла ошибка при отправке: {e}")

from lightning.pytorch.utilities.rank_zero import rank_zero_only

@rank_zero_only
def _t(self, k):
    v = self.trainer.callback_metrics.get(k)
    if v is None:
        return "—"
    if torch.is_tensor(v):
        v = v.item()
    return f"{float(v):.6f}"

def generate_tg_message(self):
    return (
            f"[{str(self.cfg.system.tb_version).upper()}]\n"
            f"Epoch: {self.current_epoch + 1}/{self.cfg.system.max_epochs}\n\n"

            f"Train:\n"
            f"g_loss   = {_t(self,'train/g_loss')}\n"
            f"loss_gan = {_t(self,'train/loss_gan')}\n"
            f"loss_fm  = {_t(self,'train/loss_fm')}\n"
            f"loss_l1  = {_t(self,'train/loss_l1')}\n"
            f"loss_fft = {_t(self,'train/loss_fft')}\n"
            f"loss_d   = {_t(self,'train/loss_d')}\n\n"

            f"Val:\n"
            f"loss_l1 = {_t(self,'val/loss_l1')}\n"
            f"psnr    = {_t(self,'val/psnr')}\n"
            f"ssim    = {_t(self,'val/ssim')}\n"
            f"lpips   = {_t(self,'val/lpips')}\n"
            f"ergas   = {_t(self,'val/ergas')}\n"
            f"sam     = {_t(self,'val/sam')}\n\n"

            f"AdaptiveLoss:\n"
            f"w_l1  = {_t(self,'loss/w_l1')}\n"
            f"w_fft = {_t(self,'loss/w_fft')}\n\n"

            f"Feats:\n"
            f"d_real = {_t(self,'feats/d_real_mean')}\n"
            f"d_fake = {_t(self,'feats/d_fake_mean')}\n\n"

            f"LR:\n"
            f"gen = {_t(self,'lr/g')}\n"
            f"dis = {_t(self,'lr/d')}\n\n"

            f"Fusion:\n"
            f"w_hfcf   = {_t(self,'fusion/w_hfcf')}\n"
            f"spat_std = {_t(self,'fusion/spatial_std')}"
        )