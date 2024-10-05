@echo off
setlocal

set "start_model=%1"
set "start_processing=false"

rem Define dataset types
set "dataset_types=anime manga"

for /f "delims=" %%m in (models.txt) do (
    if "%%m"=="%start_model%" (
        set "start_processing=true"
    )
    if "%start_processing%"=="true" (
        for %%d in (%dataset_types%) do (
            echo Generating embeddings for model: %%m on dataset: %%d
            python sbert.py --model "%%m" --type "%%d"
        )
    )
)