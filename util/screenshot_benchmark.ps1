<#
.SYNOPSIS
Runs the benchmark, waits for it to settle, takes a screenshot of the primary display and closes it.

.EXAMPLE
util\screenshot_benchmark.ps1 -Out shots\after.png

Run from the repository root (the app expects it as working directory). The app opens a borderless window
covering the primary display, so it must stay in front during the capture.
Use this rather than RenderDoc for final frame images: RenderDoc saves swapchain images with its
"undefined contents" pattern.
#>
param(
	[Parameter(Mandatory)][string]$Out,
	[int]$WaitSeconds = 25,
	[string]$Exe = "build\Debug\brutus-vulkan.exe",
	[string]$Arguments = "--benchmark"
)
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class ForegroundWindow { [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd); }
"@

$process = Start-Process -FilePath (Resolve-Path $Exe).Path -ArgumentList $Arguments -WorkingDirectory (Get-Location) -PassThru
Start-Sleep -Seconds $WaitSeconds
$process.Refresh()
if ($process.HasExited) { throw "App exited early with code $($process.ExitCode)" }
[ForegroundWindow]::SetForegroundWindow($process.MainWindowHandle) | Out-Null
Start-Sleep -Seconds 2

$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$directory = Split-Path -Parent $Out
if ($directory) { New-Item -ItemType Directory -Force $directory | Out-Null }
$bitmap.Save($Out)
$graphics.Dispose(); $bitmap.Dispose()
Stop-Process -Id $process.Id -Force
"saved $Out ($($bounds.Width)x$($bounds.Height))"
