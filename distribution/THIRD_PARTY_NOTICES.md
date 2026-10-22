# Third-party notices — Subtitle-AUTO 0.1.0

Subtitle-AUTO Quickstart's original source is MIT licensed (see QUICKSTART_LICENSE.txt).
Independent third-party libraries and downloadable model weights retain their terms.

Component | License | Upstream
---|---|---
faster-whisper 1.2.1 | MIT | https://github.com/SYSTRAN/faster-whisper
CTranslate2 4.7.1 | MIT | https://github.com/OpenNMT/CTranslate2
PyAV 17.1.0 (binding) | BSD-3-Clause | https://github.com/PyAV-Org/PyAV
FFmpeg, dynamically linked LGPL-only build | LGPL v3 or later | https://ffmpeg.org/legal.html
Hugging Face Hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub
Tokenizers | Apache-2.0 | https://github.com/huggingface/tokenizers
NumPy | BSD-3-Clause plus bundled notices | https://github.com/numpy/numpy
Whisper Tiny/Base/Small models | MIT, downloaded separately | https://huggingface.co/Systran/faster-whisper-tiny
PyInstaller | GPL with special bootloader linking exception | https://pyinstaller.org/en/stable/license.html

## FFmpeg LGPL redistribution

The PyAV binding was independently rebuilt against LGPL-only FFmpeg shared DLLs.
The build checks forbid --enable-gpl, --enable-nonfree and GPL codec options.
Runtime inspection verifies that each loaded FFmpeg library reports LGPL terms.
This build uses --enable-version3 and therefore follows LGPL v3 or later.

The exact corresponding FFmpeg sources, PyAV sources, build scripts, build flags,
and wheel provenance are downloadable alongside this application as:

  Subtitle-AUTO-LGPL-Sources-0.1.0.zip

Source download location:
https://github.com/p4147691/SUBTITLE-AUTO/releases/tag/v0.1.0

The FFmpeg library DLLs are dynamically linked; rebuild instructions are included
in the source archive to help use a compatible modified LGPL library. Reverse
engineering for debugging permitted changes to these libraries is permitted
under the LGPL. No FFmpeg GPL/x264/x265 builds are intended to be redistributed.

The README, this document, QUICKSTART_LICENSE.txt, and PRIVACY.md are bundled
with the Windows installer and Portable ZIP. Neither model weights nor private
user files are bundled. Other developer engines are excluded from Quickstart.
