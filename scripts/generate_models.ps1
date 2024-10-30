param(
    [string]$StartModel = "toobi/anime"
)

$startProcessing = $false

# Define dataset types
$datasetTypes = @("anime", "manga")

Get-Content models.txt | ForEach-Object {
    if ($_ -eq $StartModel) {
        $startProcessing = $true
    }

    if ($startProcessing) {
        foreach ($datasetType in $datasetTypes) {
            Write-Host "Generating embeddings for model: $_ on dataset: $datasetType"
            python ./src/sbert.py --model "$_" --type "$datasetType"
        }
    }
}