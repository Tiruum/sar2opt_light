import os
from dotenv import load_dotenv
import telebot

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
                print(f"❌ Файл не найден: {image_path}")
                return

            # Открываем файл и отправляем фото
            with open(image_path, 'rb') as photo:
                bot.send_photo(TELEGRAM_RECIEVER_USER_ID, photo, caption=message)

        # Если передан только текст (без изображения)
        elif message:
            # Отправляем текстовое сообщение
            bot.send_message(TELEGRAM_RECIEVER_USER_ID, message)

    except Exception as e:
        print(f"❌ Произошла ошибка при отправке: {e}")