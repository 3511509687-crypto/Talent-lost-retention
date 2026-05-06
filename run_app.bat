@echo off
setlocal

if defined HR_MODEL_PYTHON (
    "%HR_MODEL_PYTHON%" app.py
) else (
    python app.py
)

endlocal
