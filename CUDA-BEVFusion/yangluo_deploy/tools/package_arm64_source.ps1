param(
    [string]$Output = "yangluo_arm64_source.tar.gz"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$solutionRoot = (Resolve-Path (Join-Path $repoRoot "..")).Path
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $repoRoot $Output
}

$required = @(
    "CUDA-BEVFusion\CMakeLists.txt",
    "dependencies\stb",
    "dependencies\dlpack",
    "dependencies\pybind11",
    "libraries\cuOSD\src",
    "libraries\3DSparseConvolution\libspconv\include\spconv\engine.hpp",
    "libraries\3DSparseConvolution\libspconv\lib\aarch64_cuda13.0\libspconv.so"
)
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $solutionRoot $relative))) {
        throw "Missing required source/dependency: $relative"
    }
}

$exclude = @(
    "--exclude=CUDA-BEVFusion/.git",
    "--exclude=CUDA-BEVFusion/.agents",
    "--exclude=CUDA-BEVFusion/.idea",
    "--exclude=CUDA-BEVFusion/.pptx_build",
    "--exclude=CUDA-BEVFusion/.sdk_cache",
    "--exclude=CUDA-BEVFusion/.tmp_patent_case",
    "--exclude=CUDA-BEVFusion/outputs",
    "--exclude=CUDA-BEVFusion/yangluogang",
    "--exclude=CUDA-BEVFusion/*.bag",
    "--exclude=CUDA-BEVFusion/*.jsonl",
    "--exclude=CUDA-BEVFusion/*.tar.gz",
    "--exclude=CUDA-BEVFusion/yangluo_deploy/generated/*.tar.gz",
    "--exclude=CUDA-BEVFusion/senseauto_active"
)
$inputs = @(
    "CUDA-BEVFusion",
    "dependencies",
    "libraries/cuOSD/src",
    "libraries/3DSparseConvolution/libspconv"
)

$outputDirectory = Split-Path -Parent $outputPath
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

Push-Location $solutionRoot
try {
    & tar.exe -czf $outputPath @exclude @inputs
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$archive = Get-Item -LiteralPath $outputPath
$hash = Get-FileHash -LiteralPath $outputPath -Algorithm SHA256
Write-Output ("ARM64_SOURCE_PACKAGE_OK path={0} bytes={1}" -f $archive.FullName, $archive.Length)
Write-Output ("SHA256={0}" -f $hash.Hash.ToLowerInvariant())
