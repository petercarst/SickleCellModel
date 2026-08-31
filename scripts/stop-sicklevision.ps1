<#
Stops the FastAPI bridge, the Laravel dev server, and the TF Serving
container. Leaves MySQL and Docker Desktop running, since other local
projects on this machine may depend on them.
#>

$logDir = "C:\xampp\htdocs\SickleCellClassification\logs"

foreach ($name in "fastapi", "laravel") {
    $pidFile = "$logDir\$name.pid"
    if (Test-Path $pidFile) {
        $procId = Get-Content $pidFile
        # /T kills the whole process tree in case anything spawned a child process
        taskkill /PID $procId /T /F *> $null
        Remove-Item $pidFile
        Write-Host "[OK] Stopped $name (PID $procId)"
    } else {
        Write-Host "[--] No PID file for $name (already stopped, or started outside this script)"
    }
}

try {
    docker stop sickle-tfserving *> $null
    Write-Host "[OK] Stopped TF Serving container"
} catch {
    Write-Host "[--] TF Serving container was not running"
}

Write-Host ""
Write-Host "(MySQL and Docker Desktop left running - stop them manually if desired)"
