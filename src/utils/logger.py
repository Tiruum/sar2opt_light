import sys
import datetime
from colorama import init, Fore, Style
from tqdm import tqdm

# Инициализация colorama
init(autoreset=True)

class Logger:
    """
    Простой логгер с цветами (Colorama) и emoji-префиксами.
    Уровни: debug, info, success, warning, error.
    Цвета приближены к Tailwind CSS (основные оттенки).
    """
    LEVELS = {
        'debug':   {'color': Fore.MAGENTA, 'emoji': '🐞'},
        'info':    {'color': Fore.CYAN,    'emoji': 'ℹ️'},
        'success': {'color': Fore.GREEN,   'emoji': '✅'},
        'warning': {'color': Fore.YELLOW,  'emoji': '⚠️'},
        'error':   {'color': Fore.RED,     'emoji': '❌'},
    }

    def __init__(self, name: str = None, stream=sys.stdout):
        """
        name: необязательное имя логгера (будет выводиться в скобках)
        stream: куда писать (stdout или stderr)
        """
        self.name = name
        self.stream = stream

    def _log(self, level: str, message: str):
        now = datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        lvl = self.LEVELS.get(level, self.LEVELS['info'])
        parts = [
            Style.DIM + now,
            lvl['color'] + ' ' + lvl['emoji'] + ' ' + level.upper(),
        ]
        if self.name:
            parts.append(Style.BRIGHT + f"[{self.name}]")
        parts.append(Style.NORMAL + str(message))
        text = " ".join(parts) + Style.RESET_ALL
        print(text, file=self.stream)

    def debug(self, message: str):
        self._log('debug', message)

    def info(self, message: str):
        self._log('info', message)

    def success(self, message: str):
        self._log('success', message)

    def warning(self, message: str):
        self._log('warning', message)

    def error(self, message: str):
        self._log('error', message)


# Пример использования:
if __name__ == "__main__":
    logger = Logger(name="SAR2OPT")
    logger.debug("Начинаем отладку сети")
    logger.info("Загрузка датасета")
    logger.success("Модель успешно обучена")
    logger.warning("LR слишком высок, возможно расходимость")
    logger.error("Ошибка при чтении чекпоинта")