<#
  One-shot environment setup.

      powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

  Creates .venv with uv, installs CUDA torch plus the diffusion stack, and
  verifies that torch can actually see the GPU.
#>

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "`n=== vidgen setup ===`n" -ForegroundColor Cyan

# --- uv ------------------------------------------------------------------- #
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "uv not found. Installing..." -ForegroundColor Yellow
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Write-Host "uv: $((uv --version))" -ForegroundColor Green

# --- venv ----------------------------------------------------------------- #
if (-not (Test-Path ".venv")) {
    Write-Host "`nCreating virtual environment (Python 3.11)..." -ForegroundColor Cyan
    uv venv --python 3.11
} else {
    Write-Host "`n.venv already exists - reusing it." -ForegroundColor Green
}

# --- dependencies --------------------------------------------------------- #
Write-Host "`nInstalling dependencies (torch is ~2.5 GB, this takes a while)..." -ForegroundColor Cyan
uv pip install -r requirements.txt

# The project itself, without re-resolving the stack we just installed.
Write-Host "`nRegistering the vidgen package..." -ForegroundColor Cyan
uv pip install -e . --no-deps

# --- verify --------------------------------------------------------------- #
# Run through the venv interpreter rather than `uv run`: bare `uv run` re-syncs
# the environment against pyproject.toml, and on OneDrive-backed folders a locked
# file mid-sync can leave a package half-uninstalled.
Write-Host "`nVerifying the CUDA build..." -ForegroundColor Cyan
$verify = @'
import torch
print("torch          ", torch.__version__)
print("cuda available ", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device         ", p.name)
    print("vram           ", f"{p.total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: no CUDA device visible - generation would run on CPU.")
'@
$verifyFile = Join-Path $env:TEMP "vidgen_verify.py"
Set-Content -Path $verifyFile -Value $verify -Encoding utf8
& .\.venv\Scripts\python.exe $verifyFile
Remove-Item $verifyFile -Force

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Start the app with:  .\.venv\Scripts\streamlit.exe run app.py`n" -ForegroundColor Cyan
