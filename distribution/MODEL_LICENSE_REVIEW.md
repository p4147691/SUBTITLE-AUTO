# Model and runtime license review for Subtitle-AUTO 0.1.0 Quickstart

Verified release scope: faster-whisper Whisper Tiny/Base/Small CPU/GPU transcription.
Models are downloaded explicitly on first use; weights are not redistributed.

- Whisper Tiny/Base/Small: MIT model licenses, separate user-initiated download.
- faster-whisper / CTranslate2: MIT notices present.
- PyAV binding: BSD-3-Clause, independently rebuilt against LGPL-only FFmpeg
  shared libraries. The PyPI GPL-enabled wheel is rejected by binary inspection.
- FFmpeg linked binaries: LGPL v3+, corresponding upstream commit, build flags
  and source bundle are distributed alongside the application.
- PyInstaller runtime bootloader: GPL with special linking exception.
- Personal media never uploaded by application; only model download requests go to Hugging Face.

The original developer-only SenseVoice, Qwen, FSMN-VAD and OCR features are
not included in this Quickstart 0.1.0 binary. Their independent license reviews
remain out of scope for this first Windows distribution.

Release gating requires matching CPU Windows install/transcribe/uninstall CI
and physical NVIDIA CUDA packaged EXE transcription and clean process exit,
plus SHA256 checksums of all release files.
