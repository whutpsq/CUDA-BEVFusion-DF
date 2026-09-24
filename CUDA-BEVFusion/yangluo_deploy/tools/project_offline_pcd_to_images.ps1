param(
    [Parameter(Mandatory = $true)]
    [string]$FrameDir,

    [Parameter(Mandatory = $true)]
    [string]$CameraCalibration,

    [int]$DrawStride = 4
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function Read-PcdXyzi {
    param([string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $prefixLength = [Math]::Min(4096, $bytes.Length)
    $prefix = [System.Text.Encoding]::ASCII.GetString($bytes, 0, $prefixLength)
    $marker = "DATA binary`n"
    $markerIndex = $prefix.IndexOf($marker)
    if ($markerIndex -lt 0) {
        throw "Only binary PCD is supported: $Path"
    }

    $header = $prefix.Substring(0, $markerIndex + $marker.Length)
    $match = [regex]::Match($header, "(?m)^POINTS\s+(\d+)\s*$")
    if (-not $match.Success) {
        throw "POINTS entry is missing from $Path"
    }

    $count = [int]$match.Groups[1].Value
    $dataOffset = $markerIndex + $marker.Length
    if (($bytes.Length - $dataOffset) -ne ($count * 16)) {
        throw "Expected packed x/y/z/intensity float32 PCD data"
    }

    return [pscustomobject]@{
        Bytes = $bytes
        Count = $count
        DataOffset = $dataOffset
    }
}

function Get-DepthColor {
    param([double]$Depth)

    $ratio = [Math]::Max(0.0, [Math]::Min(1.0, $Depth / 60.0))
    if ($ratio -lt 0.5) {
        $local = $ratio * 2.0
        $red = 0
        $green = [int](255.0 * $local)
        $blue = [int](255.0 * (1.0 - $local))
    }
    else {
        $local = ($ratio - 0.5) * 2.0
        $red = [int](255.0 * $local)
        $green = [int](255.0 * (1.0 - $local))
        $blue = 0
    }
    return [System.Drawing.Color]::FromArgb(235, $red, $green, $blue)
}

function Save-Jpeg {
    param(
        [System.Drawing.Bitmap]$Bitmap,
        [string]$Path,
        [long]$Quality = 95
    )

    $codec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() |
        Where-Object { $_.MimeType -eq "image/jpeg" } |
        Select-Object -First 1
    $parameters = New-Object System.Drawing.Imaging.EncoderParameters(1)
    $parameters.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter(
        [System.Drawing.Imaging.Encoder]::Quality,
        $Quality
    )
    try {
        $Bitmap.Save($Path, $codec, $parameters)
    }
    finally {
        $parameters.Dispose()
    }
}

function Project-Overlay {
    param(
        [string]$ImagePath,
        [string]$OutputPath,
        [object]$Calibration,
        [object]$Cloud,
        [string]$Label,
        [int]$Stride
    )

    $intrinsic = @($Calibration.intrinsics)
    $extrinsic = @($Calibration.extrinsics)
    $distortion = @($Calibration.distortion)
    if ($intrinsic.Count -ne 9 -or $extrinsic.Count -ne 16 -or $distortion.Count -lt 5) {
        throw "Malformed calibration for $Label"
    }

    $k1 = [double]$distortion[0]
    $k2 = [double]$distortion[1]
    $p1 = [double]$distortion[2]
    $p2 = [double]$distortion[3]
    $k3 = [double]$distortion[4]
    $k4 = if ($distortion.Count -gt 5) { [double]$distortion[5] } else { 0.0 }
    $k5 = if ($distortion.Count -gt 6) { [double]$distortion[6] } else { 0.0 }
    $k6 = if ($distortion.Count -gt 7) { [double]$distortion[7] } else { 0.0 }
    $fx = [double]$intrinsic[0]
    $fy = [double]$intrinsic[4]
    $cx = [double]$intrinsic[2]
    $cy = [double]$intrinsic[5]

    $source = [System.Drawing.Image]::FromFile($ImagePath)
    $bitmap = New-Object System.Drawing.Bitmap($source.Width, $source.Height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.DrawImage($source, 0, 0, $source.Width, $source.Height)
    $source.Dispose()
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighSpeed

    $positiveDepth = 0
    $insideImage = 0
    $drawn = 0
    try {
        for ($index = 0; $index -lt $Cloud.Count; $index += 1) {
            $base = $Cloud.DataOffset + $index * 16
            $x = [double][System.BitConverter]::ToSingle($Cloud.Bytes, $base)
            $y = [double][System.BitConverter]::ToSingle($Cloud.Bytes, $base + 4)
            $z = [double][System.BitConverter]::ToSingle($Cloud.Bytes, $base + 8)

            # The raw PCD payload is RFU.  The supplied extrinsic maps RFU
            # vehicle coordinates directly into OpenCV camera coordinates.
            $cameraX = [double]$extrinsic[0] * $x + [double]$extrinsic[1] * $y + [double]$extrinsic[2] * $z + [double]$extrinsic[3]
            $cameraY = [double]$extrinsic[4] * $x + [double]$extrinsic[5] * $y + [double]$extrinsic[6] * $z + [double]$extrinsic[7]
            $cameraZ = [double]$extrinsic[8] * $x + [double]$extrinsic[9] * $y + [double]$extrinsic[10] * $z + [double]$extrinsic[11]
            if ($cameraZ -le 0.5) {
                continue
            }
            $positiveDepth += 1

            $xn = $cameraX / $cameraZ
            $yn = $cameraY / $cameraZ
            $r2 = $xn * $xn + $yn * $yn
            $r4 = $r2 * $r2
            $r6 = $r4 * $r2
            $denominator = 1.0 + $k4 * $r2 + $k5 * $r4 + $k6 * $r6
            if ([Math]::Abs($denominator) -lt 1.0e-12) {
                continue
            }
            $radial = (1.0 + $k1 * $r2 + $k2 * $r4 + $k3 * $r6) / $denominator
            $xd = $xn * $radial + 2.0 * $p1 * $xn * $yn + $p2 * ($r2 + 2.0 * $xn * $xn)
            $yd = $yn * $radial + $p1 * ($r2 + 2.0 * $yn * $yn) + 2.0 * $p2 * $xn * $yn
            $u = $fx * $xd + $cx
            $v = $fy * $yd + $cy
            if ([double]::IsNaN($u) -or [double]::IsInfinity($u) -or
                [double]::IsNaN($v) -or [double]::IsInfinity($v)) {
                continue
            }
            if ($u -lt 0.0 -or $u -ge $bitmap.Width -or $v -lt 0.0 -or $v -ge $bitmap.Height) {
                continue
            }
            $insideImage += 1
            if (($insideImage % [Math]::Max(1, $Stride)) -ne 0) {
                continue
            }

            $brush = New-Object System.Drawing.SolidBrush((Get-DepthColor $cameraZ))
            try {
                $graphics.FillEllipse($brush, [single]($u - 2.0), [single]($v - 2.0), 4.0, 4.0)
            }
            finally {
                $brush.Dispose()
            }
            $drawn += 1
        }

        $background = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(190, 0, 0, 0))
        $foreground = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
        $font = New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Bold)
        try {
            $graphics.FillRectangle($background, 0, 0, $bitmap.Width, 54)
            $graphics.DrawString(
                "$Label | visible=$insideImage | drawn=$drawn | blue=near red=far",
                $font,
                $foreground,
                12,
                12
            )
        }
        finally {
            $font.Dispose()
            $foreground.Dispose()
            $background.Dispose()
        }
    }
    finally {
        $graphics.Dispose()
    }

    try {
        Save-Jpeg -Bitmap $bitmap -Path $OutputPath -Quality 95
    }
    finally {
        $bitmap.Dispose()
    }

    return [ordered]@{
        label = $Label
        source_image = [System.IO.Path]::GetFullPath($ImagePath)
        output = [System.IO.Path]::GetFullPath($OutputPath)
        positive_depth_points = $positiveDepth
        projected_inside_image = $insideImage
        drawn_points = $drawn
    }
}

$resolvedFrameDir = (Resolve-Path -LiteralPath $FrameDir).Path
$resolvedCalibration = (Resolve-Path -LiteralPath $CameraCalibration).Path
$pcdPath = Join-Path $resolvedFrameDir "lidar_frame_5s_raw_rfu.pcd"
$cam5Path = Join-Path $resolvedFrameDir "cam5_frame_5s_synced.jpg"
$cam10Path = Join-Path $resolvedFrameDir "cam10_frame_5s_synced.jpg"
foreach ($required in @($pcdPath, $cam5Path, $cam10Path)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required input: $required"
    }
}

$calibrationText = Get-Content -Raw -LiteralPath $resolvedCalibration
$calibrationText = [regex]::Replace($calibrationText, "/\*.*?\*/", "", [System.Text.RegularExpressions.RegexOptions]::Singleline)
$calibration = $calibrationText | ConvertFrom-Json
$byId = @{}
foreach ($item in $calibration.camera_params) {
    $byId[[string]$item.desc] = $item
}
foreach ($cameraId in @("0", "1", "5", "10")) {
    if (-not $byId.ContainsKey($cameraId)) {
        throw "Calibration ID $cameraId is missing"
    }
}

$cloud = Read-PcdXyzi -Path $pcdPath
$results = @()
$results += Project-Overlay -ImagePath $cam5Path -OutputPath (Join-Path $resolvedFrameDir "cam5_projection_rfu_calib0.jpg") -Calibration $byId["0"] -Cloud $cloud -Label "cam5 | RFU | calibration 0 (verified)" -Stride $DrawStride
$results += Project-Overlay -ImagePath $cam5Path -OutputPath (Join-Path $resolvedFrameDir "cam5_projection_rfu_calib1.jpg") -Calibration $byId["1"] -Cloud $cloud -Label "cam5 | RFU | calibration 1 (comparison)" -Stride $DrawStride
$results += Project-Overlay -ImagePath $cam10Path -OutputPath (Join-Path $resolvedFrameDir "cam10_projection_rfu_calib10.jpg") -Calibration $byId["10"] -Cloud $cloud -Label "cam10 | RFU | calibration 10 (verified)" -Stride $DrawStride
$results += Project-Overlay -ImagePath $cam10Path -OutputPath (Join-Path $resolvedFrameDir "cam10_projection_rfu_calib5.jpg") -Calibration $byId["5"] -Cloud $cloud -Label "cam10 | RFU | calibration 5 (comparison)" -Stride $DrawStride

$report = [ordered]@{
    point_cloud = [System.IO.Path]::GetFullPath($pcdPath)
    point_coordinate_frame = "RFU raw payload"
    point_count = $cloud.Count
    camera_calibration = $resolvedCalibration
    projection_model = "OpenCV rational pinhole with raw RFU-to-camera extrinsics"
    draw_stride = $DrawStride
    results = $results
}
$reportPath = Join-Path $resolvedFrameDir "projection_report_5s.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding utf8

$results | ForEach-Object {
    Write-Output ("PROJECTION_OK label={0} inside={1} drawn={2} output={3}" -f $_.label, $_.projected_inside_image, $_.drawn_points, $_.output)
}
Write-Output "report=$reportPath"
Write-Output "OFFLINE_PROJECTION_COMPLETE"
