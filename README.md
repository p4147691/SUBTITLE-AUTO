# Subtitle-AUTO 0.1.0 — 무료 Windows 자막 생성기

이 배포판은 Whisper 기반 기본 자막 생성 기능을 제공하는 첫 Windows 버전입니다.
**SenseVoice·Qwen·OCR·제목 자동 변경 등 기존 개발자 기능은 이번 배포판에 포함되지 않습니다.**
그 기능은 기존 개발용 소스와 로컬 실행 환경에서 계속 사용할 수 있습니다.

## 다운로드 및 시작

GitHub Releases에서 Windows용 Subtitle-AUTO-Setup-0.1.0.exe 또는
Subtitle-AUTO-Portable-0.1.0.zip을 다운로드합니다.

- Setup.exe: 설치 후 시작 메뉴에서 Subtitle-AUTO 자막 생성 (GPU 우선)을 실행
- Portable ZIP: 압축을 해제하고 Subtitle-AUTO.exe 실행
- 음성 또는 영상 파일을 선택합니다.
- Whisper Tiny, Base, Small 중 모델을 선택합니다. 처음 사용하면 모델을 인터넷에서 내려받습니다.
- 장치: auto(기본, NVIDIA GPU 우선·CPU 대체), cuda(GPU 전용), cpu(CPU 전용)
- 생성 버튼을 누르면 같은 폴더에 UTF-8 SRT가 생깁니다.
- 기존 SRT는 절대 덮어쓰지 않습니다. 영상·음원 원본도 보존합니다.

## 실행 조건

Windows x64, 적절한 저장 공간과 인터넷(최초 모델 다운로드)이 필요합니다.
CPU 모드에는 별도 CUDA 드라이버가 필요하지 않으며, 일부 매우 오래된 CPU는 지원되지 않을 수 있습니다.
NVIDIA GPU를 이용하려면 호환 드라이버·CUDA/cuDNN 환경이 필요할 수 있습니다.
GPU 초기화가 실패하면 auto 모드는 CPU로 전환하며, cuda 전용은 실패 원인을 보고합니다.
Python과 FFmpeg의 별도 수동 설치는 기본 자막 생성기에 필요하지 않습니다.
기본 환경과 모델별 성능은 PC 사양에 따라 달라집니다.

## 개인정보·라이선스

음성과 영상은 로컬에서 처리됩니다. 모델 파일 다운로드 시 Hugging Face에 접속합니다.
배포 파일에는 모델 가중치·개인 파일·비밀번호·사용자 설정이 포함되지 않습니다.
자세한 사항은 PRIVACY.md, THIRD_PARTY_NOTICES.md, QUICKSTART_LICENSE.txt를 확인하세요.
FFmpeg LGPL 구성과 재빌드에 필요한 대응 소스는 GitHub Releases의 Subtitle-AUTO-LGPL-Sources-0.1.0.zip으로 제공합니다.
모델 캐시는 사용자 LOCALAPPDATA/Subtitle-AUTO/models에 저장됩니다.
설치 제거 뒤에도 사용자가 내려받은 모델은 남을 수 있습니다.

## 탐색기 Alt+S (다음 버전 개발 중)

파일 탐색기에서 영상/음성 파일을 선택하고 Alt+S를 누르면 앱에서 확인한 후 자막을 생성합니다.
아무것도 선택하지 않으면 탐색기에서 열린 폴더의 영상·음성을 대상으로 합니다.
선택한 폴더가 있으면 그 폴더의 파일만 대상으로 하며 하위 폴더는 자동 검색하지 않습니다.
기존 SRT는 건너뛰고, 한번에 최대 200개까지만 처리합니다.
다른 프로그램에서 Alt+S를 쓰고 있으면 등록하지 않고 버튼을 사용할 수 있습니다.
앱 실행 중(최소화 포함)에만 작동하며, Windows 로그인 시 최소화 실행은 사용자가
설정에서 직접 동의한 경우에만 등록합니다. 기존 클립보드를 수정하지 않습니다.

## 선택형 SenseVoice·Qwen (개발 중)

Whisper는 기본 설치 상태를 유지합니다.
SenseVoiceSmall (약 0.95GB)은 공식 FunASR 모델 라이선스 고지와 별도
FunASR/VAD 구동 환경이 필요합니다.
Qwen3-ASR-1.7B (약 4.1GB)는 Apache 2.0이며, 시간 정렬에는
Qwen3 ForcedAligner (약 1.85GB)가 추가로 필요할 수 있습니다.
모델 파일만 내려받고 엔진이 설치됐다고 잘못 표시하지 않기 위해,
두 고급 엔진의 자동 설치 및 실행 버튼은 통합 런타임 E2E가 완료될 때까지
비활성 상태입니다. 기본 설치에서 다운로드나 별도 GPU 라이브러리
설치는 실행하지 않습니다.

## 개발자 고급 기능 (배포판 범위 밖)

기존 Subtitle-AUTO 전체 기능에는 별도 Python 가상환경, 모델, GPU, FFmpeg가 필요합니다.
고급 엔진은 배포판 0.1.0의 설치·검증 대상이 아니며 사용자 PC에서 자동 설치되지 않습니다.
다음 명령은 기존 개발자 설정에서만 사용하세요:

    py -3.12 distribution/bootstrap.py --check

기존 config.json과 로컬 모델을 덮어쓰거나 이동하지 않습니다.

## 지원 및 개선

https://github.com/p4147691/SUBTITLE-AUTO/issues
