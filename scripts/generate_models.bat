@echo off
setlocal

rem Set default start model if not provided
if "%1"=="" (
    set "start_model=toobi/anime"
) else (
    set "start_model=%1"
)
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
            python ./src/sbert.py --model "%%m" --type "%%d"
        )
    )
)