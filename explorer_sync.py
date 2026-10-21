from __future__ import annotations

import ctypes
import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def _norm_path(value: str | os.PathLike[str]) -> str:
    try:
        return os.path.normcase(os.path.normpath(str(Path(value))))
    except Exception:
        return os.path.normcase(os.path.normpath(str(value)))


def _window_path(window) -> str:
    try:
        document = window.Document
        folder = document.Folder
        current = folder.Self.Path
        if current:
            return str(current)
    except Exception:
        pass

    try:
        location_url = str(window.LocationURL or "")
    except Exception:
        return ""

    if not location_url.lower().startswith("file:"):
        return ""

    parsed = urlparse(location_url)
    path = unquote(parsed.path or "")
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    elif len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path.replace("/", "\\")


def sync_folder_rename(old_folder: str | os.PathLike[str], new_folder: str | os.PathLike[str]) -> bool:
    if os.name != "nt" or not old_folder or not new_folder:
        return False

    old_text = str(old_folder)
    new_text = str(new_folder)

    try:
        ctypes.windll.shell32.SHChangeNotify(
            0x00020000,  # SHCNE_RENAMEFOLDER
            0x0005,      # SHCNF_PATHW
            ctypes.c_wchar_p(old_text),
            ctypes.c_wchar_p(new_text),
        )
    except Exception:
        return False

    old_norm = _norm_path(old_text)

    try:
        import win32com.client

        shell = win32com.client.Dispatch("Shell.Application")
        for window in shell.Windows():
            try:
                current = _window_path(window)
                if current and _norm_path(current) == old_norm:
                    window.Navigate2(new_text)
                    try:
                        window.Refresh()
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception:
        pass

    return True
