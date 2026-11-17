param([switch]$KeepContainer)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$containerStatus = docker ps --filter name=camera_processor_ws --format "{{.Status}}"
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo consultar Docker."
}

if ($containerStatus -like "*Up*") {
    $launchPid = docker exec camera_processor_ws bash -lc "cat /tmp/camera_processor_launch.pid 2>/dev/null"
    $launchPid = ([string]$launchPid).Trim()
    if ($launchPid -match '^\d+$') {
        $launchCommand = docker exec camera_processor_ws ps -p $launchPid -o args=
        if ($LASTEXITCODE -ne 0 -or $launchCommand -notmatch "ros2 launch camera_processor") {
            $launchPid = ""
        }
    }

    if (-not $launchPid) {
        $launchPid = docker exec camera_processor_ws bash -lc "pgrep -f '^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch camera_processor ' | head -n 1"
        $launchPid = ([string]$launchPid).Trim()
    }

    $targetPid = $launchPid
    if (-not $targetPid) {
        $targetPid = docker exec camera_processor_ws bash -lc "pgrep -f '^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 bag record ' | head -n 1"
        $targetPid = ([string]$targetPid).Trim()
    }

    if ($targetPid -match '^\d+$') {
        Write-Host "Cerrando ROS 2 y finalizando la grabacion..."
        docker exec camera_processor_ws kill -INT $targetPid
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo enviar la señal de cierre al proceso ROS 2 ($targetPid)."
        }

        $deadline = (Get-Date).AddSeconds(60)
        do {
            docker exec camera_processor_ws bash -lc "kill -0 $targetPid 2>/dev/null"
            $processRunning = $LASTEXITCODE -eq 0
            if ($processRunning) {
                Start-Sleep -Seconds 1
            }
        } while ($processRunning -and (Get-Date) -lt $deadline)

        if ($processRunning) {
            throw "ROS 2 no terminó en 60 segundos; el contenedor se mantiene activo para proteger la bolsa."
        }

        docker exec camera_processor_ws rm -f /tmp/camera_processor_launch.pid
        Write-Host "Grabacion finalizada."
    }
}

if (-not $KeepContainer) {
    docker-compose down
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo detener el contenedor Docker."
    }
}

$true
