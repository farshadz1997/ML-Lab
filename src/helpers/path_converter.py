import sys, os, traceback
from datetime import datetime


def resource_path(relative_path: str,  exc_path: bool = False) -> str:
    """Get absolute path for resource, works for dev and for PyInstaller"""
    try:
        running_file_path = sys.executable
        if hasattr(sys, "_MEIPASS"):
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
            if exc_path:
                base_path = os.path.dirname(running_file_path)
        elif os.path.splitext(running_file_path)[1] == ".exe" and os.path.isfile(os.path.join(os.path.dirname(running_file_path), "flutter_windows.dll")):
            base_path = os.path.dirname(running_file_path)
        else:
            base_path = os.getcwd()
    except Exception as e:
        save_error_path = os.path.join(os.getcwd(), "errors.txt")
        tb = e.__traceback__
        tb_str = traceback.format_tb(tb)
        error = "\n".join(tb_str).strip() + f"\n{e}"
        with open(save_error_path, "a") as f:
            f.write(f"\n-------------------{datetime.now()}-------------------\r\n")
            f.write(f"{error}\n")
            return
    else:
        return os.path.join(base_path, relative_path)
