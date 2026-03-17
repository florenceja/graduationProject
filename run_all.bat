@echo off
chcp 65001 >nul
echo ============================================
echo   EDANE 全流程实验一键运行
echo ============================================
echo.

echo [1/6] 合成数据实验...
python src/edane_full_pipeline.py --mode synthetic --quantize
echo.

echo [2/6] Reddit 样本实验...
python src/edane_full_pipeline.py --mode file --dataset-preset reddit_sample --snapshots 6 --quantize
echo.

echo [3/6] Amazon2M 样本实验...
python src/edane_full_pipeline.py --mode file --dataset-preset amazon2m_sample --snapshots 4 --quantize
echo.

echo [4/6] MAG 样本实验...
python src/edane_full_pipeline.py --mode file --dataset-preset mag_sample --snapshots 6 --quantize
echo.

echo [5/6] Twitter 样本实验...
python src/edane_full_pipeline.py --mode file --dataset-preset twitter_sample --snapshots 6 --quantize
echo.

echo [6/6] Amazon-3M 样本实验...
python src/edane_full_pipeline.py --mode file --dataset-preset amazon3m_sample --snapshots 6 --quantize
echo.

echo ============================================
echo   全部实验完成！结果在 outputs/ 目录下
echo ============================================
pause
