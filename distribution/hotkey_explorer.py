"""Opt-in OS-level Alt+S, routed only from active File Explorer windows.

No clipboard automation, no global keyboard hook, no privilege elevation.
The hotkey exists only while the GUI is open (including minimized).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import queue

HOTKEY_ID = 0x5341
WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_NOREPEAT = 0x4000
VK_S = 0x53
PM_REMOVE = 0x0001
EXPLORER_CLASSES = frozenset({"CabinetWClass", "ExploreWClass"})
MAX_FILES_PER_BATCH = 200
MEDIA_SUFFIXES = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav",
    ".m4a", ".flac", ".ogg", ".aac",
})


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_ssize_t),
        ("time", ctypes.c_ulong),
        ("pt", POINT),
        ("lPrivate", ctypes.c_ulong),
    ]


@dataclass(frozen=True)
class ExplorerSnapshot:
    folder: Path
    selected: tuple[Path, ...] = ()


def current_explorer_snapshot() -> ExplorerSnapshot | None:
    """Read only the foreground Explorer's COM selection; never touch clipboard."""
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetClassNameW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)
    user32.GetClassNameW.restype = ctypes.c_int
    hwnd = int(user32.GetForegroundWindow() or 0)
    if not hwnd:
        return None
    class_buffer = ctypes.create_unicode_buffer(128)
    if not user32.GetClassNameW(hwnd, class_buffer, 128):
        return None
    if class_buffer.value not in EXPLORER_CLASSES:
        return None
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    try:
        shell = win32com.client.Dispatch("Shell.Application")
        for window in shell.Windows():
            try:
                if int(window.HWND) != hwnd:
                    continue
                document = window.Document
                folder = Path(str(document.Folder.Self.Path))
                if not folder.is_dir():
                    return None
                selected = tuple(Path(str(item.Path)) for item in document.SelectedItems())
                return ExplorerSnapshot(folder=folder, selected=selected)
            except (AttributeError, OSError, ValueError, TypeError):
                continue
        return None
    finally:
        pythoncom.CoUninitialize()


def collect_media(snapshot: ExplorerSnapshot, max_files: int = MAX_FILES_PER_BATCH) -> tuple[Path, ...]:
    """Selected files/folders when selected; otherwise direct files in current folder.

    A hard upper bound prevents accidentally submitting a whole disk.
    Folder scanning is nonrecursive: Alt+S never silently walks subdirectories.
    """
    if max_files < 1:
        raise ValueError("max_files must be >= 1")
    candidates = snapshot.selected or (snapshot.folder,)
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.is_symlink():
            continue
        if candidate.is_dir():
            children = sorted(candidate.iterdir(), key=lambda f: f.name.casefold())
        elif candidate.is_file():
            children = [candidate]
        else:
            continue
        for child in children:
            if child.is_symlink() or not child.is_file() or child.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            key = os.path.normcase(str(child.resolve()))
            if key in seen:
                continue
            seen.add(key)
            if len(result) >= max_files:
                raise ValueError(f"MEDIA_BATCH_LIMIT_{max_files}")
            result.append(child)
    return tuple(result)


def classify_targets(items: tuple[Path, ...]) -> tuple[tuple[Path, ...], int]:
    """Never overwrite existing SRTs; omit inputs already processed."""
    accepted = tuple(x for x in items if not x.with_suffix(".srt").exists())
    return accepted, len(items) - len(accepted)


class WindowsAltS:
    """RegisterHotKey is owned by a dedicated message-loop thread.

    Tk may consume thread-window messages before a timer-based PeekMessage poll,
    so a dedicated GetMessage loop delivers hotkeys via a safe in-process queue.
    """

    def __init__(self) -> None:
        self.registered = False
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._events: queue.Queue[bool] = queue.Queue()
        self._ready = threading.Event()
        self._stop = threading.Event()

    def _message_loop(self) -> None:
        api = ctypes.windll.user32
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentThreadId.restype = ctypes.c_ulong
        self._thread_id = int(kernel.GetCurrentThreadId())
        api.PeekMessageW.argtypes = (ctypes.POINTER(MSG), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
        api.PeekMessageW.restype = ctypes.c_bool
        api.GetMessageW.argtypes = (ctypes.POINTER(MSG), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint)
        api.GetMessageW.restype = ctypes.c_int
        api.RegisterHotKey.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint)
        api.RegisterHotKey.restype = ctypes.c_bool
        api.UnregisterHotKey.argtypes = (ctypes.c_void_p, ctypes.c_int)
        api.UnregisterHotKey.restype = ctypes.c_bool
        msg = MSG()
        # Establish the message queue before another thread can request shutdown.
        api.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        self.registered = bool(api.RegisterHotKey(None, HOTKEY_ID, MOD_ALT | MOD_NOREPEAT, VK_S))
        self._ready.set()
        if not self.registered:
            return
        try:
            while not self._stop.is_set():
                code = api.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if code <= 0:
                    break
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    self._events.put(True)
        finally:
            api.UnregisterHotKey(None, HOTKEY_ID)
            self.registered = False

    def register(self) -> bool:
        if os.name != "nt":
            return False
        if self._thread and self._thread.is_alive():
            return self.registered
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._message_loop, name="SubtitleAutoAltS", daemon=True)
        self._thread.start()
        # Fast bounded confirmation; don't silently steal an existing binding.
        return self._ready.wait(1.5) and self.registered

    def poll(self) -> bool:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return False

    def unregister(self) -> None:
        self._stop.set()
        if self._thread_id and os.name == "nt":
            api = ctypes.windll.user32
            api.PostThreadMessageW.argtypes = (ctypes.c_ulong, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t)
            api.PostThreadMessageW.restype = ctypes.c_bool
            api.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)  # WM_QUIT
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._thread_id = None
        self.registered = False
        while self.poll():
            pass
