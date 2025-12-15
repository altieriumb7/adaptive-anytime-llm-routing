param(
  [Parameter(Mandatory=$true)][string]$BatchId,
  [string]$Out = "data\batch_anytime_train_out.jsonl",
  [string]$Err = "data\batch_anytime_train_err.jsonl",
  [int]$EverySeconds = 30
)

# Assumes your python venv is active and scripts/fetch_batch_files.py loads .env
Write-Host "Watching batch: $BatchId"
Write-Host "Polling every $EverySeconds seconds..."
Write-Host ""

while ($true) {
  # Fetch current status via python (prints status + counts + file ids)
  python scripts\fetch_batch_files.py --batch_id $BatchId --out $Out --err $Err

  # If output exists, stop
  if (Test-Path $Out) {
    Write-Host ""
    Write-Host "✅ Output downloaded: $Out"
    break
  }

  # If error file exists, report its size (can appear only at end)
  if (Test-Path $Err) {
    $lines = (Get-Content $Err | Measure-Object -Line).Lines
    Write-Host "⚠ Error file present: $Err (lines=$lines)"
  }

  Start-Sleep -Seconds $EverySeconds
}
