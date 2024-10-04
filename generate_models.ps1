Get-Content models.txt | ForEach-Object {
    Write-Host "Generating embeddings for model: $_"
    python sbert.py --model "$_"
}