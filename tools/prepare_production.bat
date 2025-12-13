@echo off
echo ===================================================
echo APEX TRADE AI - PRODUCTION PREPARATION
echo ===================================================

echo [1/4] Repairing Features (Schema Enforcement)...
python tools/repair_features.py

echo [2/4] Retraining All Models...
python tools/train_multiframe.py

echo [3/4] Validating System...
python tools/validate_models.py

echo [4/4] System Ready.
echo Run 'python tools/predict_all.py' to start dashboard.
pause
