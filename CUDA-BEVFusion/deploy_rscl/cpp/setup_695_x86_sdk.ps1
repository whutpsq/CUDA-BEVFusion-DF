param(
  [string]$ReleaseRoot = "H:\df_code\695_x86",
  [string]$OutputRoot = ".sdk_cache\695_x86",
  [switch]$WithThirdParty
)

$ErrorActionPreference = "Stop"

function Expand-FirstPackage {
  param(
    [Parameter(Mandatory = $true)][string]$Pattern,
    [Parameter(Mandatory = $true)][string]$Label
  )

  $pkg = Get-ChildItem -LiteralPath $ReleaseRoot -Filter $Pattern | Select-Object -First 1
  if (-not $pkg) {
    throw "Cannot find $Label package in $ReleaseRoot with pattern $Pattern"
  }

  Write-Host "Extracting $Label package: $($pkg.Name)"
  tar -xf $pkg.FullName -C $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Expand-FirstPackage -Pattern "senseauto-rscl-*.tar.gz" -Label "senseauto-rscl"
Expand-FirstPackage -Pattern "senseauto-msgs-*.tar.gz" -Label "senseauto-msgs"

if ($WithThirdParty) {
  Expand-FirstPackage -Pattern "senseauto-3rdparty-*.tar.gz" -Label "senseauto-3rdparty"
} else {
  Write-Host "Skipping senseauto-3rdparty. Re-run with -WithThirdParty if runtime linker reports missing capnp/kj/protobuf/glog/gflags libraries."
}

Write-Host "SDK cache prepared at $OutputRoot"
