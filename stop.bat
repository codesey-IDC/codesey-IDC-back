@echo off
rem start.bat 으로 띄운 서비스 3개와 대시보드를 끈다.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo [IDC] 서비스를 종료합니다...

rem 1) start.bat 이 띄운 창을 제목으로 찾아 그 안의 python 까지 함께 종료
for %%T in (IDC-twin IDC-forecast IDC-mpc IDC-dashboard) do (
    taskkill /FI "WINDOWTITLE eq %%T*" /T /F >nul 2>nul
)

rem 2) 다른 방법으로 띄운 경우를 위해 포트를 쓰는 프로세스도 종료
for %%P in (8001 8002 8003 8501) do (
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr /R /C:":%%P .*LISTENING"') do (
        taskkill /PID %%I /T /F >nul 2>nul
    )
)

set "LEFT="
for %%P in (8001 8002 8003 8501) do (
    netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul && (
        echo [주의] 포트 %%P 가 아직 사용 중입니다.
        set "LEFT=1"
    )
)
if not defined LEFT echo [IDC] 모두 종료했습니다.
endlocal
