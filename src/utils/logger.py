import sys
import datetime
import os
import inspect
from pathlib import Path
from colorama import init, Fore, Style
from omegaconf import OmegaConf

# Включаем поддержку UTF-8 в Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

init(autoreset=True)

class Logger:
    LEVELS = {
        'debug':   {'color': Fore.MAGENTA, 'emoji': '🐞'},
        'info':    {'color': Fore.CYAN,    'emoji': 'ℹ️'},
        'success': {'color': Fore.GREEN,   'emoji': '✅'},
        'warning': {'color': Fore.YELLOW,  'emoji': '⚠️'},
        'error':   {'color': Fore.RED,     'emoji': '❌'},
    }

    def __init__(self, name: str = None, stream=sys.stdout, cfg_path: str = 'src/models/cfrwd/config.yaml'):
        self.cfg = OmegaConf.load(cfg_path)
        self.name = name if name else str(self.cfg.system.tb_version).upper()
        self.stream = stream
        self.logged_lines = set()
        # Определяем корень проекта (папку, содержащую src)
        self.project_root = self._find_project_root()

    def _find_project_root(self) -> Path:
        """Находит корневую папку проекта (содержащую src)"""
        # Начинаем от текущего файла и поднимаемся вверх, пока не найдем папку src
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            if (parent / 'src').exists():
                return parent
        # Если не нашли, возвращаем текущую папку
        return current_path.parent

    def _get_relative_path(self, absolute_path: str) -> str:
        """Преобразует абсолютный путь в относительный от корня проекта"""
        try:
            absolute_path_obj = Path(absolute_path).resolve()
            # Пытаемся получить относительный путь от корня проекта
            relative_path = absolute_path_obj.relative_to(self.project_root)
            return str(relative_path)
        except ValueError:
            # Если путь не находится внутри корня проекта, возвращаем только имя файла
            return Path(absolute_path).name

    def _get_caller_info(self) -> tuple:
        """Возвращает (относительный_путь, номер_строки) вызывающего кода"""
        try:
            frame = inspect.stack()[3]
            absolute_path = frame.filename
            relative_path = self._get_relative_path(absolute_path)
            return (relative_path, frame.lineno)
        finally:
            del frame

    def _log(self, level: str, message: str, show_lineno: bool = False, once: bool = False):
        now = datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        lvl = self.LEVELS.get(level, self.LEVELS['info'])
        
        parts = [
            Style.DIM + now,
            lvl['color'] + ' ' + lvl['emoji'] + ' ' + level.upper(),
        ]
        
        if self.name:
            if show_lineno:
                filename, lineno = self._get_caller_info()
                parts.append(Style.BRIGHT + f"[{self.name} {filename}:{lineno}]")
            else:
                parts.append(Style.BRIGHT + f"[{self.name}]")
        
        if once:
            filename, lineno = self._get_caller_info()
            unique_id = (filename, lineno, message)
            if unique_id in self.logged_lines:
                return
            self.logged_lines.add(unique_id)
        parts.append(Style.NORMAL + str(message))
        text = " ".join(parts) + Style.RESET_ALL
        print(text, file=self.stream, flush=True)

    def debug(self, message: str, show_lineno: bool = True, show: bool = None, once: bool = False):
        if show is None:
            try:
                show = self.cfg.system.debug
            except Exception:
                show = False
        if show:
            self._log('debug', message, show_lineno, once)

    def info(self, message: str, show_lineno: bool = False):
        self._log('info', message, show_lineno)

    def success(self, message: str, show_lineno: bool = False):
        self._log('success', message, show_lineno)

    def warning(self, message: str, show_lineno: bool = False):
        self._log('warning', message, show_lineno)

    def error(self, message: str, show_lineno: bool = False):
        self._log('error', message, show_lineno)


# Пример использования:
if __name__ == "__main__":
    logger = Logger(name="SAR2OPT")
    logger.debug("Начинаем отладку сети")  # Покажет номер этой строки
    logger.info("Загрузка датасета")
    logger.success("Модель успешно обучена")
    logger.warning("LR слишком высок, возможно расходимость")
    logger.error("Ошибка при чтении чекпоинта")