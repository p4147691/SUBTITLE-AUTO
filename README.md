# Subtitle-AUTO — Windows 배포 준비판
이 저장소는 현재 **개발자 소스**입니다. 설치 없이 즉시 실행되는 완성형 앱으로 오해하지 마세요.

## 설치 전 점검 (읽기 전용)
Python 3.12, ffmpeg, ffprobe, NVIDIA 드라이버가 필요하며 기존 엔진은 GPU 기반입니다.

    py -3.12 distribution/bootstrap.py --check

## 초기 개인 설정
    py -3.12 distribution/bootstrap.py --init-config

기존 config.json은 덮어쓰지 않습니다. 모델 및 Python 가상환경은 자동 설치되지 않습니다.
모델 선택 다운로드는 명시적 라이선스 동의를 요구하며, 검증되지 않은 저장소 ID는 다운로드를 차단합니다.
모델 파일과 사용자의 음성·자막은 GitHub에 업로드하지 않습니다.

## 배포 게이트
Portable ZIP/Setup.exe 초안은 CI에서 별도로 생성합니다. **정식 공개 배포는 실제 깨끗한 Windows에서
모델 설치·자막 생성·제거·재설치 E2E가 통과할 때까지 보류합니다.**
