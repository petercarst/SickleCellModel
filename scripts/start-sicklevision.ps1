<#
Starts everything SickleVision needs: MySQL (XAMPP), Docker Desktop + the
TF Serving container, the FastAPI bridge, and the Laravel dev server.
Safe to re-run - each step checks whether its service is already up first.
#>

$root   = "C:\xampp\htdocs\SickleCellClassification"
$logDir = "$root\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Test-Port($port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

Write-Host "== SickleVision startup ==" -ForegroundColor Cyan

# --- 1. MySQL ---
if (Test-Port 3306) {
    Write-Host "[OK]   MySQL already running on :3306"
} else {
    Write-Host "[..]   Starting MySQL..."
    Start-Process "C:\xampp\mysql_start.bat" -WindowStyle Minimized
    $tries = 0
    while (-not (Test-Port 3306) -and $tries -lt 30) { Start-Sleep -Seconds 2; $tries++ }
    if (Test-Port 3306) { Write-Host "[OK]   MySQL is up" }
    else { Write-Host "[FAIL] MySQL did not start within 60s" -ForegroundColor Red }
}

# --- 2. Docker Desktop + TF Serving container ---
function Test-Docker { try { docker info *> $null; return $true } catch { return $false } }

$dockerUp = Test-Docker
if (-not $dockerUp) {
    Write-Host "[..]   Starting Docker Desktop (can take ~60s)..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    $tries = 0
    while (-not $dockerUp -and $tries -lt 24) {
        Start-Sleep -Seconds 5
        $dockerUp = Test-Docker
        $tries++
    }
}

if ($dockerUp) {
    Write-Host "[OK]   Docker is up"
    $exists  = (docker ps -a --filter "name=^sickle-tfserving$" --format "{{.Names}}") -eq "sickle-tfserving"
    $running = (docker ps      --filter "name=^sickle-tfserving$" --format "{{.Names}}") -eq "sickle-tfserving"
    if ($running) {
        Write-Host "[OK]   TF Serving container already running"
    } elseif ($exists) {
        Write-Host "[..]   Starting TF Serving container..."
        docker start sickle-tfserving | Out-Null
    } else {
        Write-Host "[..]   Creating TF Serving container..."
        docker run -d --name sickle-tfserving -p 8501:8501 `
            -v "$root\models\sickle-cell:/models/sickle-cell" `
            -e MODEL_NAME=sickle-cell tensorflow/serving | Out-Null
    }
    Start-Sleep -Seconds 3
} else {
    Write-Host "[FAIL] Docker did not start in time - TF Serving will be unavailable" -ForegroundColor Red
}

# --- 3. FastAPI bridge ---
if (Test-Port 8010) {
    Write-Host "[OK]   FastAPI already running on :8010"
} else {
    Write-Host "[..]   Starting FastAPI bridge..."
    $p = Start-Process python -ArgumentList "api\tf_serving.py" -WorkingDirectory $root `
        -RedirectStandardOutput "$logDir\fastapi.log" -RedirectStandardError "$logDir\fastapi.err.log" `
        -WindowStyle Hidden -PassThru
    $p.Id | Out-File "$logDir\fastapi.pid"
}

# --- 4. Laravel dev server ---
# Uses PHP's built-in server directly (not "artisan serve", which spawns this
# as an unrelated child process that survives if the parent is killed).
if (Test-Port 8001) {
    Write-Host "[OK]   Laravel already running on :8001"
} else {
    Write-Host "[..]   Starting Laravel dev server..."
    $p = Start-Process "C:\xampp\php\php.exe" -ArgumentList "-S","127.0.0.1:8001","-t","public" -WorkingDirectory "$root\sickleVision" `
        -RedirectStandardOutput "$logDir\laravel.log" -RedirectStandardError "$logDir\laravel.err.log" `
        -WindowStyle Hidden -PassThru
    $p.Id | Out-File "$logDir\laravel.pid"
}

Start-Sleep -Seconds 3
Write-Host ""
Write-Host "== Status ==" -ForegroundColor Cyan
Write-Host ("MySQL          : " + $(if (Test-Port 3306) {"UP"} else {"DOWN"}))
Write-Host ("TF Serving     : " + $(if (Test-Port 8501) {"UP"} else {"DOWN"}))
Write-Host ("FastAPI bridge : " + $(if (Test-Port 8010) {"UP"} else {"DOWN"}))
Write-Host ("Laravel app    : " + $(if (Test-Port 8001) {"UP"} else {"DOWN"}))
Write-Host ""
Write-Host "SickleVision: http://127.0.0.1:8001" -ForegroundColor Green
