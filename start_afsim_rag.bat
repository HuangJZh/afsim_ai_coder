@echo off
echo AFSIM RAG代码生成系统启动中...
echo.

REM 运行系统
echo.
echo 启动AFSIM RAG系统...
echo 1. 索引文档模式
echo 2. Web界面模式
echo 3. 命令行模式
echo 4. 退出
echo.

set /p choice="请选择模式 (1-4): "

if "%choice%"=="1" (
    python main_fixed.py --mode index
) else if "%choice%"=="2" (
    echo 启动Web界面...
    echo 请访问: http://localhost:7860
    python main_fixed.py --mode web
) else if "%choice%"=="3" (
    python main_fixed.py --mode cli
) else (
    echo 退出
)

pause