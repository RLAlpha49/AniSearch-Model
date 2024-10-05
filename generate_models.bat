@echo off
setlocal

set "start_model=%1"
set "start_processing=false"

for /f "delims=" %%m in (models.txt) do (
    if "%%m"=="%start_model%" (
        set "start_processing=true"
    )
    if "%start_processing%"=="true" (
        echo Generating embeddings for model: %%m
        python sbert.py --model "%%m"
    )
)