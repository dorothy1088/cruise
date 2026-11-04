부산항 ETS 대시보드 - Streamlit Community Cloud 배포안

이 폴더에는 원본 파일 2개를 Cloud용으로 정리한 버전이 있습니다.
- app.py: 사용자가 업로드한 부산항_ETS_전체분석 (3)(1).py 기반 (신규 실적 갱신/백테스트/시드 안정성)
- app_v2.py: 사용자가 업로드한 부산항_ETS_전체분석 (2)(1).py 기반 (기존 2026 검증/2027 비교)
- requirements.txt: 두 앱 모두의 의존성 (Python 3.12 권장)
- smoke_test.py: 서버 및 의존성 테스트용

1. 압축을 풀고 streamlit_deploy 폴더 전체를 기존 GitHub cruise 저장소의 최상위 폴더에 올립니다.
   (전체 폴더를 올려야 합니다. zip 파일만 GitHub에 올리면 실행되지 않습니다.)
2. Streamlit Community Cloud에서 앱을 새로 배포할 때
   Repository: dorothy1088/cruise
   Branch: main
   Main file path: streamlit_deploy/app.py
   Advanced settings > Python version: 3.12
   를 선택하세요.
   (예전 버전을 실행하려면 streamlit_deploy/app_v2.py를 선택하세요.)
3. 본 저장소 루트에 uv.lock과 pyproject.toml이 있더라도,
   앱 엔트리포인트 폴더에 있는 streamlit_deploy/requirements.txt가 먼저 탐색됩니다.
4. 새 배포가 실패할 경우 먼저 로그의 마지막 Traceback을 확인하세요.
   'Spinning up manager process'에서 멈추고 Python 실행 로그가 없다면
   Cloud 자체의 머신 시작 문제일 수 있습니다.
   비교 진단: streamlit_deploy/smoke_test.py 를 엔트리포인트로 새 배포해 보세요.
5. 화면이 열리면 CSV를 업로드하고 분석을 실행합니다. 업로드 전에는 ETS 계산이 시작되지 않습니다.

변경점:
- Cloud가 이미 Streamlit 서버를 실행하므로 로컬 자동 서버 실행/브라우저 실행 wrapper 제거
- PyCharm 로컬 실행 버튼 대신 python -m streamlit run <파일경로> 사용
- app.py 내 '이 컴퓨터에 저장' 표현을 Cloud 서버 임시 저장으로 명확화
- 기존 ETS 수식, 시드, 경로 수, 결측 처리, 지표 산출 로직은 변경하지 않음

주의:
- GitHub에 원본 데이터 파일이나 비밀번호를 공개로 올리지 마세요. 필요하면 앱 안에서 CSV 업로드.
- Cloud 서버의 임시 파일은 재시작/재배포 시 사라질 수 있습니다. 예측 기록은 JSON 다운로드로 보관하세요.
- 배포 후 정확한 수치 검증에는 사용자의 CSV 파일과 실행 로그가 추가로 필요합니다.
- 처음 앱이 열리지 않는 문제와 CSV 업로드 후 계산이 느린 문제는 구분해서 진단해야 합니다.
