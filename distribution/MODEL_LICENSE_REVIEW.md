# 모델/런타임 라이선스 검토 — 배포 게이트

- SenseVoiceSmall: https://huggingface.co/FunAudioLLM/SenseVoiceSmall (모델 카드의 license: model-license). 재배포 조건 원문 확인 전 가중치 **동봉 금지**.
- Qwen3-ASR 및 ForcedAligner: https://github.com/QwenLM/Qwen3-ASR (공식 다운로드 경로 안내). 각 배포 태그의 정확한 모델별 라이선스·notice 및 변경사항 확인 필요.
- Whisper: https://huggingface.co/Systran/faster-whisper-large-v3 (MIT). 현재 사용 중인 large-v3-turbo와 다른 저장소이므로 turbo 출처 및 라이선스 확인 필요.
- FSMN-VAD: 로컬 저장소와 동일한 정확한 업스트림 모델 ID 및 라이선스 미확정. 다운로드 자동화 차단.
- FFmpeg: 배포할 바이너리의 LGPL/GPL 빌드 옵션·고지·소스 제공 의무 확인 필요.
- CUDA/PyTorch/ctranslate2/torch wheels: 바이너리 재배포 라이선스 및 NVIDIA 호환 드라이버 조건 검토 필요.
- 사용자 미디어·자막은 로컬 처리 전제이며 다운로드 동의 외에 원격 전송 없음. 개인정보·민감 경로를 로그로 노출하지 않도록 추후 감사 필요.

판정: **NOT_RELEASE_READY**. 소스 전용 프리뷰만 CI에서 빌드. 사용자용 공식 Release 금지.
