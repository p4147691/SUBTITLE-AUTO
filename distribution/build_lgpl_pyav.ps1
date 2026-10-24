# Rebuild PyAV 17.1.0 against a SHA-256-pinned, LGPL-only, shared FFmpeg 8.1 build.
# The PyPI PyAV binary wheel is NOT used as a release binary.
param([string]$OutputDirectory = "distribution/lgpl-build-output")
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $root $OutputDirectory
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$archive = Join-Path $destination "ffmpeg-lgpl-shared-8.1.zip"
# Pin a dated upstream release asset AND its published SHA-256 digest. The
# moving /latest URL changes bytes frequently and must not be used with a
# static security checksum. A replaced fixed asset still fails closed.
$uri = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-21-13-06/ffmpeg-n8.1.3-20-gc8de997cb1-win64-lgpl-shared-8.1.zip"
$expected = "2eec333253518c6deb0a8b6dff6d615e03f254baa1b04f44356de2725e5fb60f"
if (!(Test-Path $archive)) {
  Invoke-WebRequest -Uri $uri -OutFile $archive -MaximumRedirection 8
}
$actual = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "LGPL_FFMPEG_SHA256_MISMATCH expected=$expected actual=$actual" }
Write-Output "FFMPEG_IMMUTABLE_SHA256_PASS $actual"
$unpack = Join-Path $destination "ffmpeg-lgpl"
if (!(Test-Path $unpack)) { Expand-Archive -Path $archive -DestinationPath $unpack }
$roots = Get-ChildItem -Path $unpack -Directory -Recurse | Where-Object {
    (Test-Path (Join-Path $_.FullName "bin/ffmpeg.exe")) -and
    (Test-Path (Join-Path $_.FullName "lib")) -and
    (Test-Path (Join-Path $_.FullName "include"))
}
if (@($roots).Count -ne 1) { throw "UNEXPECTED_FFMPEG_ROOT_COUNT $(@($roots).Count)" }
$ffDir = $roots[0].FullName
$ffBin = Join-Path $ffDir "bin"
$ffexe = Join-Path $ffBin "ffmpeg.exe"
$versionOutput = & $ffexe -version | Out-String
if ($LASTEXITCODE -ne 0) { throw "FFMPEG_BUILD_VERSION_FAILED" }
$versionOutput | Set-Content -Path (Join-Path $destination "FFMPEG_VERSION.txt") -Encoding UTF8
Write-Output "FFMPEG_VERSION: $($versionOutput.Split([Environment]::NewLine)[0])"
$buildinfo = & $ffexe -buildconf | Out-String
if ($LASTEXITCODE -ne 0) { throw "FFMPEG_BUILD_CONFIG_FAILED" }
if ($buildinfo -match "--enable-gpl" -or $buildinfo -match "--enable-nonfree" -or $buildinfo -match "--enable-libx264" -or $buildinfo -match "--enable-libx265") {
  throw "FFMPEG_GPL_OR_NONFREE_FLAGS_DETECTED"
}
if ($buildinfo -notmatch "--enable-shared") { throw "FFMPEG_IS_NOT_SHARED" }
$buildinfo | Set-Content -Path (Join-Path $destination "FFMPEG_BUILD_CONFIGURATION.txt") -Encoding UTF8
Write-Output "FFMPEG_LGPL_ONLY_SHARED_CONFIGURATION_PASS"
$env:PATH = "$ffBin;$env:PATH"
$pyavSourceSha = "97241598792f4c980c32458187154c7c999d68b3"
$wheelhouse = Join-Path $destination "wheelhouse"
New-Item -ItemType Directory -Path $wheelhouse -Force | Out-Null
$manifest = Join-Path $wheelhouse "LGPL_WHEEL_CACHE_V1.json"
$existingWheels = @(Get-ChildItem -Path $wheelhouse -Filter "av-17.1.0-*.whl" -File)
$cacheHit = Test-Path $manifest
if ($cacheHit) {
  # GitHub cache is only a performance hint, never a release gate. Verify
  # the restored wheel SHA, Python ABI, pinned FFmpeg hash, and PyAV source
  # commit BEFORE installing it. Always run the LGPL runtime gate later.
  & python (Join-Path $PSScriptRoot "lgpl_wheel_cache.py") verify --wheelhouse $wheelhouse --ffmpeg-sha $expected --pyav-sha $pyavSourceSha
  if ($LASTEXITCODE -ne 0) { throw "LGPL_WHEEL_CACHE_INVALID_FAIL_CLOSED" }
  $repaired = @(Get-ChildItem -Path $wheelhouse -Filter "av-17.1.0-*.whl" -File)
  if ($repaired.Count -ne 1) { throw "LGPL_CACHED_WHEEL_COUNT_INVALID" }
  Write-Output "LGPL_VERIFIED_WHEEL_CACHE_HIT"
} else {
  if ($existingWheels.Count -gt 0) { throw "LGPL_WHEEL_CACHE_INCOMPLETE_FAIL_CLOSED" }
  & python -m pip install --disable-pip-version-check --quiet "Cython==3.1.6" "setuptools>=70,<90" wheel delvewheel "numpy>=1.26,<3"
  if ($LASTEXITCODE -ne 0) { throw "PYAV_BUILD_DEPENDENCY_INSTALL_FAILED" }
  $pyav = Join-Path $destination "PyAV-v17.1.0-source"
  if (!(Test-Path $pyav)) {
    git clone --depth 1 --branch v17.1.0 https://github.com/PyAV-Org/PyAV.git $pyav
    if ($LASTEXITCODE -ne 0) { throw "PYAV_SOURCE_FETCH_FAILED" }
  }
  $sourceSha = (git -C $pyav rev-parse HEAD).Trim()
  if ($sourceSha -ne $pyavSourceSha) { throw "PYAV_SOURCE_COMMIT_MISMATCH $sourceSha" }
  Write-Output "PYAV_SOURCE_PINNED_PASS $sourceSha"
  Push-Location $pyav
  try {
    & python setup.py bdist_wheel "--ffmpeg-dir=$ffDir"
    if ($LASTEXITCODE -ne 0) { throw "PYAV_LGPL_BUILD_FAILED" }
  }
  finally { Pop-Location }
  $wheels = @(Get-ChildItem -Path (Join-Path $pyav "dist") -Filter "av-17.1.0-*.whl")
  if ($wheels.Count -ne 1) { throw "PYAV_LGPL_WHEEL_COUNT_FAILED $($wheels.Count)" }
  & python -m delvewheel repair --add-path $ffBin --wheel-dir $wheelhouse $wheels[0].FullName
  if ($LASTEXITCODE -ne 0) { throw "PYAV_LGPL_DELVEWHEEL_FAILED" }
  $repaired = @(Get-ChildItem -Path $wheelhouse -Filter "av-17.1.0-*.whl" -File)
  if ($repaired.Count -ne 1) { throw "PYAV_LGPL_REPAIRED_WHEEL_COUNT_FAILED $($repaired.Count)" }
  Write-Output "LGPL_WHEEL_CACHE_COLD_BUILD"
}
& python -m pip uninstall -y av
if ($LASTEXITCODE -ne 0) { throw "OLD_AV_UNINSTALL_FAILED" }
& python -m pip install --no-deps --force-reinstall $repaired[0].FullName
if ($LASTEXITCODE -ne 0) { throw "LGPL_AV_INSTALL_FAILED" }
& python (Join-Path $PSScriptRoot "verify_lgpl_pyav.py") --wheel $repaired[0].FullName --output $destination
if ($LASTEXITCODE -ne 0) { throw "LGPL_AV_VERIFICATION_FAILED" }
# Only cache a wheel that passed the actual imported FFmpeg LGPL checks.
if (!$cacheHit) {
  & python (Join-Path $PSScriptRoot "lgpl_wheel_cache.py") record --wheelhouse $wheelhouse --ffmpeg-sha $expected --pyav-sha $pyavSourceSha
  if ($LASTEXITCODE -ne 0) { throw "LGPL_WHEEL_CACHE_RECORD_FAILED" }
}
Write-Output "PYAV_LGPL_BINARY_GATE_PASS $($repaired[0].FullName)"
