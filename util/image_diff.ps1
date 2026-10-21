<#
.SYNOPSIS
Compares two images of the same size pixel by pixel (benchmark renders are deterministic, any difference is real).

.EXAMPLE
util\image_diff.ps1 -A before.png -B after.png -OutDir diff -IgnoreLeftPixels 490 -Crops "near,1150,330,450,420,1.5;far,490,180,1430,140,0.6"

Prints the share of changed pixels and writes diff_x8.png (absolute difference amplified 8x). Each crop
("name,x,y,width,height,scale") writes a side-by-side image, A on the left and B on the right.
-IgnoreLeftPixels skips a band on the left of the image, e.g. the ImGui panel (490 pixels wide).
#>
param(
	[Parameter(Mandatory)][string]$A,
	[Parameter(Mandatory)][string]$B,
	[Parameter(Mandatory)][string]$OutDir,
	[int]$IgnoreLeftPixels = 0,
	[string]$Crops = ""
)
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $OutDir | Out-Null

$imageA = [System.Drawing.Bitmap]::FromFile((Resolve-Path $A).Path)
$imageB = [System.Drawing.Bitmap]::FromFile((Resolve-Path $B).Path)
if ($imageA.Width -ne $imageB.Width -or $imageA.Height -ne $imageB.Height) { throw "Images have different sizes" }
$width = $imageA.Width; $height = $imageA.Height
$rect = New-Object System.Drawing.Rectangle 0, 0, $width, $height
$format = [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
$bytes = $width * $height * 4

function Read-Pixels($image) {
	$data = $image.LockBits($rect, 'ReadOnly', $format)
	$pixels = New-Object byte[] $bytes
	[System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $pixels, 0, $bytes)
	$image.UnlockBits($data)
	return , $pixels
}
$pixelsA = Read-Pixels $imageA
$pixelsB = Read-Pixels $imageB

$diff = New-Object byte[] $bytes
$changed = 0; $large = 0; $sum = 0.0; $count = 0
for ($y = 0; $y -lt $height; $y++) {
	for ($x = $IgnoreLeftPixels; $x -lt $width; $x++) {
		$i = ($y * $width + $x) * 4
		$d = [Math]::Max([Math]::Abs($pixelsA[$i] - $pixelsB[$i]), [Math]::Max([Math]::Abs($pixelsA[$i + 1] - $pixelsB[$i + 1]), [Math]::Abs($pixelsA[$i + 2] - $pixelsB[$i + 2])))
		$v = [Math]::Min(255, $d * 8)
		$diff[$i] = $v; $diff[$i + 1] = $v; $diff[$i + 2] = $v
		$diff[$i + 3] = 255
		if ($d -gt 2) { $changed++ }
		if ($d -gt 32) { $large++ }
		$sum += $d; $count++
	}
}
$output = New-Object System.Drawing.Bitmap $width, $height, $format
$data = $output.LockBits($rect, 'WriteOnly', $format)
[System.Runtime.InteropServices.Marshal]::Copy($diff, 0, $data.Scan0, $bytes)
$output.UnlockBits($data)
$output.Save((Join-Path $OutDir "diff_x8.png"))
$output.Dispose()

"pixels compared: $count"
"changed (>2/255): {0:N2}%" -f (100.0 * $changed / $count)
"large (>32/255): {0:N2}%" -f (100.0 * $large / $count)
"mean abs diff: {0:N3}/255" -f ($sum / $count)

foreach ($crop in ($Crops -split ";" | Where-Object { $_ })) {
	$name, $cx, $cy, $cw, $ch, $scale = $crop -split ","
	$cx = [int]$cx; $cy = [int]$cy; $cw = [int]$cw; $ch = [int]$ch; $scale = [double]$scale
	$sw = [int]($cw * $scale); $sh = [int]($ch * $scale)
	$side = New-Object System.Drawing.Bitmap ($sw * 2 + 10), $sh
	$graphics = [System.Drawing.Graphics]::FromImage($side)
	$graphics.Clear([System.Drawing.Color]::White)
	$graphics.InterpolationMode = 'NearestNeighbor'
	$source = New-Object System.Drawing.Rectangle $cx, $cy, $cw, $ch
	$graphics.DrawImage($imageA, (New-Object System.Drawing.Rectangle 0, 0, $sw, $sh), $source, 'Pixel')
	$graphics.DrawImage($imageB, (New-Object System.Drawing.Rectangle ($sw + 10), 0, $sw, $sh), $source, 'Pixel')
	$side.Save((Join-Path $OutDir "sbs_$name.png"))
	$graphics.Dispose(); $side.Dispose()
}
$imageA.Dispose(); $imageB.Dispose()
