"""Non-invasive source identity, SRT validation and cross-engine folder fencing.

A source being renamed is not a reason to move it back or create its old
directory. Never silently overwrite another program's subtitle.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


class MediaChangedError(RuntimeError):
    pass


class SubtitleCollisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    sample_hash: str


def _sample_file(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(65536))
        if size > 65536:
            stream.seek(max(0, size - 65536))
            digest.update(stream.read(65536))
    return digest.hexdigest()


def capture_file(path: Path) -> FileIdentity:
    path = Path(path)
    try:
        before = path.stat()
        if not path.is_file():
            raise OSError("not a regular file")
        sample = _sample_file(path, before.st_size)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns,
                before.st_ino, before.st_dev) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns,
                after.st_ino, after.st_dev):
            raise OSError("file changed during initial inspection")
        return FileIdentity(path.absolute(), after.st_size, after.st_mtime_ns,
                            after.st_ctime_ns, after.st_dev, after.st_ino, sample)
    except OSError as exc:
        raise MediaChangedError(f"SOURCE_UNAVAILABLE_OR_CHANGING:{path}") from exc


def ensure_unchanged(snapshot: FileIdentity) -> None:
    try:
        now = capture_file(snapshot.path)
    except MediaChangedError as exc:
        raise MediaChangedError(f"SOURCE_MOVED_DELETED_OR_LOCKED:{snapshot.path}") from exc
    if now != snapshot:
        raise MediaChangedError(f"SOURCE_CHANGED_DURING_PROCESSING:{snapshot.path}")


_SRT_TIMING = re.compile(
    r"^(\d{2,}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(\d{2,}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def srt_health(path: Path) -> str:
    """Return missing, valid or invalid; never rewrite or delete an invalid SRT."""
    path = Path(path)
    if not path.exists():
        return "missing"
    try:
        if not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
            return "invalid"
        data = path.read_text(encoding="utf-8-sig")
        blocks = re.split(r"\n\s*\n", data.replace("\r\n", "\n").strip())
        if not blocks:
            return "invalid"
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3 or not lines[0].strip().isdigit():
                return "invalid"
            timing = _SRT_TIMING.match(lines[1].strip())
            if timing is None or not any(t.strip() for t in lines[2:]):
                return "invalid"
            if timing.group(1) >= timing.group(2):
                return "invalid"
        return "valid"
    except (OSError, UnicodeError):
        return "invalid"


class FolderJobLock:
    """One OS byte-range lock shared across SenseVoice and Qwen daemons."""

    def __init__(self, stream):
        self._stream = stream

    def release(self):
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.lockf(stream, fcntl.LOCK_UN, 1)
        finally:
            stream.close()


def try_folder_job_lock(target: Path) -> FolderJobLock | None:
    target = Path(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    # Lock lives with the application, NOT inside the media directory:
    # keeping an open lock file inside the video folder would prevent the
    # daemon's own final folder rename on Windows.
    meta = target.stat()
    key = (f"{meta.st_dev}:{meta.st_ino}"
           if meta.st_ino else str(target.resolve()).casefold())
    lock_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    folder = Path(__file__).resolve().parent / "private" / "folder-locks"
    folder.mkdir(parents=True, exist_ok=True)
    stream = (folder / (lock_id + ".lock")).open("a+b")
    try:
        if stream.seek(0, 2) == 0:
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                stream.close()
                return None
        else:
            import fcntl
            try:
                fcntl.lockf(stream, fcntl.LOCK_EX | fcntl.LOCK_NB, 1)
            except OSError:
                stream.close()
                return None
        return FolderJobLock(stream)
    except BaseException:
        stream.close()
        raise
