@echo off
rem 서비스 3개(트윈·예측·MPC)와 대시보드를 한 번에 띄운다. 끌 때는 stop.bat
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo [IDC] 실행 준비를 확인합니다...

where python >nul 2>nul
if errorlevel 1 (
    echo [오류] python 을 찾을 수 없습니다. Python 3.10~3.12 를 설치하고 PATH 에 추가해 주세요.
    goto :fail
)

python -c "import fastapi, uvicorn, streamlit, plotly, lightgbm, httpx" >nul 2>nul
if errorlevel 1 (
    echo [오류] 필요한 패키지가 없습니다. 먼저 아래 명령을 실행해 주세요.
    echo        pip install -r requirements.txt
    goto :fail
)

rem Git LFS 없이 받으면 데이터가 수백 바이트짜리 껍데기 파일로 내려온다
set "DATA=data\synthetic_idc_90d.parquet"
if not exist "%DATA%" (
    echo [오류] 데이터 파일이 없습니다: %DATA%
    echo        git lfs install 후 git lfs pull 을 실행해 주세요.
    goto :fail
)
for %%F in ("%DATA%") do if %%~zF LSS 10000 (
    echo [오류] 데이터 파일이 Git LFS 포인터입니다: %DATA%
    echo        git lfs install 후 git lfs pull 을 실행해 주세요.
    goto :fail
)

set "BUSY="
for %%P in (8001 8002 8003 8501) do (
    netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul && (
        echo [오류] 포트 %%P 를 이미 다른 프로그램이 쓰고 있습니다.
        set "BUSY=1"
    )
)
if defined BUSY (
    echo        이전에 띄운 서비스라면 stop.bat 을 실행한 뒤 다시 시도해 주세요.
    goto :fail
)

echo [IDC] 디지털 트윈 (8001), 예측 (8002) 을 띄웁니다...
start "IDC-twin" /min cmd /k python -m uvicorn services.twin.main:app --port 8001
start "IDC-forecast" /min cmd /k python -m uvicorn services.forecast.main:app --port 8002
call :wait_for http://127.0.0.1:8001/health "디지털 트윈" || goto :fail
call :wait_for http://127.0.0.1:8002/health "예측" || goto :fail

echo [IDC] MPC (8003) 를 띄웁니다...
start "IDC-mpc" /min cmd /k python -m uvicorn services.mpc.main:app --port 8003
call :wait_for http://127.0.0.1:8003/health "MPC" || goto :fail

echo [IDC] 대시보드 (8501) 를 띄웁니다...
start "IDC-dashboard" /min cmd /k python -m streamlit run dashboard/app.py --server.port 8501 --server.headless true
call :wait_for http://127.0.0.1:8501/_stcore/health "대시보드" || goto :fail

echo.
echo [IDC] 모두 준비됐습니다. 브라우저를 엽니다: http://127.0.0.1:8501
echo       서비스 창 4개는 작업 표시줄에 최소화되어 있습니다. 끌 때는 stop.bat 을 실행하세요.
start "" http://127.0.0.1:8501
endlocal
exit /b 0


:wait_for
rem %1 = 확인할 주소, %2 = 이름. 최대 90초 기다린다.
set /a TRIES=0
:wait_loop
curl -s -f -o nul -m 2 %1 >nul 2>nul
if not errorlevel 1 (
    echo        %~2 준비 완료
    exit /b 0
)
set /a TRIES+=1
if %TRIES% GEQ 90 (
    echo [오류] %~2 서비스가 90초 안에 뜨지 않았습니다. 작업 표시줄의 해당 창에서 오류 내용을 확인해 주세요.
    exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto :wait_loop


:fail
echo.
echo [IDC] 실행하지 못했습니다. 이미 뜬 서비스는 stop.bat 으로 끌 수 있습니다.
endlocal
pause
exit /b 1
