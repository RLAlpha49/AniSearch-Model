param(
    [string]$StartModel
)

$startProcessing = $false

Get-Content models.txt | ForEach-Object {
    if ($_ -eq $StartModel) {
        $startProcessing = $true
    }

    if ($startProcessing) {
        Write-Host "Generating embeddings for model: $_"
        python sbert.py --model "$_"
    }
}