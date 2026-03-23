@echo off
chcp 65001 >nul
echo ============================================
echo   EDANE 全流程实验一键运行
echo ============================================
echo.

echo [1/2] 合成数据实验...
python src/edane_full_pipeline.py --mode synthetic --quantize
echo.

echo [2/2] OAG 固定数据集实验...
python src/prepare_datasets.py
python src/edane_full_pipeline.py --mode file --snapshots 6 --quantize
echo.

echo ============================================
echo   全部实验完成！结果在 outputs/ 目录下
echo ============================================
pause
