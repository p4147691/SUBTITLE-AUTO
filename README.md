# Subtitle-AUTO — Windows 미리보기

## 현재 일반 사용자용 기능 (GPU 우선 Quickstart)
- GitHub Actions에서 만든 Subtitle-AUTO-Setup-0.1.0-preview.exe 또는 Subtitle-AUTO-CPU-Portable-preview.zip에서 GPU 우선 간편 실행기를 이용할 수 있습니다.
- 첫 실행 시 파일과 Tiny / Base / Small 모델을 선택합니다. 선택된 모델을 Hugging Face에서 내려받으므로 인터넷 연결과 사용자 동의가 필요합니다.
- 장치 선택: 자동(GPU 우선, 초기화 실패 시 CPU) / GPU 전용 / CPU 전용. GPU 전용은 실패하더라도 조용히 CPU로 바꾸지 않습니다.\n- NVIDIA GPU가 없어도 CPU에서 동작하도록 만들며 Python이나 시스템 FFmpeg 설치 없이 단일 EXE 형태로 빌드합니다. 실제 독립 동작은 CI에서 검증합니다.
- 모델 저장 위치: 사용자 계정의 LOCALAPPDATA/Subtitle-AUTO/models. 모델은 설치 파일에 포함하지 않습니다.
- 기존 영상이나 음원은 수정하지 않으며 SRT가 이미 있으면 덮어쓰지 않습니다. 변환은 로컬에서 수행합니다.
- CPU Quickstart는 Whisper 기반 기본 음성 자막만 지원합니다. 기존 개발자 버전의 SenseVoice, Qwen, OCR, 폴더명 변환은 별도의 Python·모델·GPU 환경이 필요하며 CPU 실행기에 아직 통합되지 않았습니다.
- Portable과 Setup.exe는 0.1.0 미리보기입니다. 완전한 제품 v1.0의 대체물이 아닙니다.

## 개발자 환경
Python 3.12 / 지원 GPU / FFmpeg와 각 모델별 환경이 필요합니다.

    py -3.12 distribution/bootstrap.py --check
    py -3.12 distribution/bootstrap.py --init-config

기존 config.json은 덮어쓰지 않습니다. 정식 모델 다운로드/환경 구축은 미완료이며 검증되지 않은 모델 소스는 차단됩니다.

## 개인정보·라이선스
개인 모델·영상·음성·SRT·인증 정보는 저장소나 릴리스에 동봉하지 않습니다.
CPU 모델 Tiny/Base/Small은 공식 Systran/faster-whisper 모델을 첫 실행 때 다운로드합니다.
https://huggingface.co/Systran/faster-whisper-tiny (MIT)
https://huggingface.co/Systran/faster-whisper-small (MIT)

고급 모델 배포 라이선스와 패키지 의존 라이선스를 확인해야 정식 v1.0 배포가 가능합니다.
