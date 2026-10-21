# Subtitle-AUTO 인코딩 호환성 정책 (2026-10-21)

## 기본 정책
- Windows 시스템 로캘의 Beta: UTF-8 체크 여부와 무관하게 동작해야 한다.
- 파일 이름과 폴더 이름은 Windows 유니코드 API / Python Path 객체로 다룬다. CP949로 파일명을 변경하지 않는다.
- SRT, 설정 JSON, 로그, 프로세스 간 JSON 데이터는 명시적 UTF-8을 사용한다. 기존 SRT/미디어는 수정하지 않는다.
- 외부 프로그램의 텍스트 출력은 encoding="utf-8", errors="replace"를 명시하고, 실제 경로는 argv 배열로 전달한다.
- 콘솔은 common.configure_unicode_stdio()로 설정한다. 표현 불가능한 문자만 진단 출력에 한해 이스케이프 허용하고 원본 파일/SRT은 보존한다.
- CMD 실행기는 %~dp0 프로젝트 루트, chcp 65001, PYTHONUTF8=1, PYTHONIOENCODING=utf-8을 사용한다. Windows 시스템 전체 설정을 변경하지 않는다.
- 레거시 파일 입력 인코딩은 형식별로 분리 판정하며 임의의 디코딩으로 원본 내용을 덮어쓰지 않는다.

## 회귀 테스트
python -m unittest test_locale_modes_20261021 test_subtitle_hotfix_20261021 test_alt_s_file_safety test_split_eta_dashboard test_scene_duplicate_merge -q

test_locale_modes_20261021은 PYTHONUTF8=0/PYTHONIOENCODING=cp949와 PYTHONUTF8=1/PYTHONIOENCODING=utf-8의 자식 프로세스에서 일본어·한글 파일명 FFmpeg 및 SRT 시험을 수행한다.
이 검증은 두 실행 인코딩 모드에 대한 시험이며 시스템 로캘을 재부팅으로 전환하는 시험이 아니다. 전체 엔진 추론 검증은 별도 수행한다.

## GitHub 동기화
2026-10-21 기준: D:\PROJECTS\SUBTITLE-AUTO 에 .git 폴더가 없고 연결된 p4147691 계정에서 전용 저장소를 찾지 못했다.
기존 저장소 URL 확인 또는 새 비공개 저장소 준비 전까지 GitHub에 게시하지 않는다.
모델·가상환경·미디어·자막·민감 설정/로그는 .gitignore로 제외하고, 게시 전 파일 목록을 검증한다.
