#.\scripts\monitor_topic_hz.ps1 -DurationPerTopic 20 -OutputFile results_hz.txt
param(
    [int]$DurationPerTopic = 15,
    [string]$OutputFile = "monitor_hz_results.txt"
)

$container = "camera_processor_ws"

# Check container running
$running = & docker ps --filter "name=$container" --format "{{.Names}}"
if (-not $running) {
    Write-Error "Container '$container' is not running. Start it with: docker compose up -d"
    exit 1
}

# Topics: images first, then node topics
$images = @(
    '/camera/image_raw',
    '/person/processed_image',
    '/pose/ia/processed_image',
    '/pose_heuristic/processed_image'
)

$nodes = @(
    '/person/detections',
    '/person/detected',
    '/pose/ia/detected',
    '/pose/heuristic/detected',
    '/pose/detected'
)

# Prepare output (save results next to this script in the scripts folder)
$fullPath = Join-Path $PSScriptRoot $OutputFile
if (Test-Path $fullPath) { Remove-Item $fullPath }
Add-Content $fullPath "Monitor run: $(Get-Date -Format o)"
Add-Content $fullPath "Duration per topic: $DurationPerTopic seconds"
Add-Content $fullPath "Container: $container"
Add-Content $fullPath ""

function RunTopic($topic, $duration) {
    Write-Host "Measuring topic $topic for $duration seconds..."
    Add-Content $fullPath "=== $(Get-Date -Format o) ==="
    Add-Content $fullPath "Topic: $topic"

    try {
        $cmd = "source /opt/ros/jazzy/setup.bash && source /workspaces/ros2_ws/install/setup.bash && timeout ${duration}s ros2 topic hz $topic"
        $output = & docker exec -i $container bash -lc $cmd 2>&1
    } catch {
        $output = "ERROR running docker exec: $_"
    }

    # Normalize output
    if ($output -is [array]) {
        $output = $output -join "`n"
    }

    Add-Content $fullPath $output
    Add-Content $fullPath ""
}

# Run image topics first
foreach ($t in $images) { RunTopic $t $DurationPerTopic }

# Then node topics
foreach ($t in $nodes) { RunTopic $t $DurationPerTopic }

Write-Host "Monitoring complete. Results saved to $fullPath"
Add-Content $fullPath "Monitoring complete: $(Get-Date -Format o)"

exit 0
