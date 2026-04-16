param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "help",
        "setup",
        "install",
        "data",
        "models",
        "models-econ",
        "models-predictive",
        "all",
        "all-from-raw",
        "dashboard",
        "clean"
    )]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $RepoRoot "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Write-Usage {
    Write-Host "ASEAN Policy Dashboard PowerShell Tasks"
    Write-Host ""
    Write-Host "Setup"
    Write-Host "  .\scripts\dev.ps1 setup"
    Write-Host ""
    Write-Host "Pipeline"
    Write-Host "  .\scripts\dev.ps1 data"
    Write-Host "  .\scripts\dev.ps1 models"
    Write-Host "  .\scripts\dev.ps1 models-econ"
    Write-Host "  .\scripts\dev.ps1 models-predictive"
    Write-Host "  .\scripts\dev.ps1 all"
    Write-Host "  .\scripts\dev.ps1 all-from-raw"
    Write-Host ""
    Write-Host "App"
    Write-Host "  .\scripts\dev.ps1 dashboard"
    Write-Host ""
    Write-Host "Maintenance"
    Write-Host "  .\scripts\dev.ps1 clean"
}

function Ensure-Venv {
    if (Test-Path $VenvPython) {
        return
    }

    Write-Host "Creating virtual environment in venv"

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv venv
        if ($LASTEXITCODE -eq 0 -and (Test-Path $VenvPython)) {
            return
        }

        & py -3 -m venv venv
        if ($LASTEXITCODE -eq 0 -and (Test-Path $VenvPython)) {
            return
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv venv
        if ($LASTEXITCODE -eq 0 -and (Test-Path $VenvPython)) {
            return
        }
    }

    throw "Unable to create a virtual environment. Install Python 3, then rerun setup."
}

function Invoke-VenvPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    Ensure-Venv
    & $VenvPython @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $VenvPython $($Args -join ' ')"
    }
}

function Setup-Environment {
    Ensure-Venv
    Write-Host "Installing dependencies"
    Invoke-VenvPython -Args @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-VenvPython -Args @("-m", "pip", "install", "-r", "requirements.txt")
    Write-Host ""
    Write-Host "Setup complete."
    Write-Host "Activate with: .\venv\Scripts\Activate.ps1"
}

function Ensure-Ready {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "No virtual environment found. Running setup first."
        Setup-Environment
    }
}

Push-Location $RepoRoot
try {
    switch ($Target) {
        "help" {
            Write-Usage
        }
        "setup" {
            Setup-Environment
        }
        "install" {
            Setup-Environment
        }
        "data" {
            Ensure-Ready
            Invoke-VenvPython -Args @("scripts/run_pipeline.py", "--stage", "data")
        }
        "models" {
            Ensure-Ready
            Invoke-VenvPython -Args @("scripts/run_pipeline.py", "--stage", "models")
        }
        "models-econ" {
            Ensure-Ready
            Invoke-VenvPython -Args @("scripts/run_pipeline.py", "--stage", "models", "--skip-predictive")
        }
        "models-predictive" {
            Ensure-Ready
            Invoke-VenvPython -Args @("scripts/run_pipeline.py", "--stage", "models", "--skip-ols", "--skip-fixed-effects")
        }
        "all" {
            Ensure-Ready
            Invoke-VenvPython -Args @("scripts/run_pipeline.py", "--stage", "all")
        }
        "all-from-raw" {
            Ensure-Ready
            Invoke-VenvPython -Args @("scripts/run_pipeline.py", "--stage", "all", "--build-from-raw-json")
        }
        "dashboard" {
            Ensure-Ready
            Invoke-VenvPython -Args @("-m", "streamlit", "run", "app/streamlit_app.py")
        }
        "clean" {
            Get-ChildItem -Path . -Directory -Filter "__pycache__" -Recurse -Force | ForEach-Object {
                Remove-Item -Path $_.FullName -Recurse -Force
            }
            Get-ChildItem -Path . -File -Filter "*.pyc" -Recurse -Force | ForEach-Object {
                Remove-Item -Path $_.FullName -Force
            }
            Write-Host "Removed Python cache files."
        }
        default {
            Write-Usage
        }
    }
}
finally {
    Pop-Location
}
