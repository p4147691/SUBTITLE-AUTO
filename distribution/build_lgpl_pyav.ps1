# Rebuild PyAV 17.1.0 against a SHA-256-pinned, LGPL-only, shared FFmpeg 8.1 build.
# The PyPI PyAV binary wheel is NOT used as a release binary.
param([string]$OutputDirectory = "distribution/lgpl-build-output")
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $root $OutputDirectory
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$archive = Join-Path $destination "ffmpeg-lgpl-shared-8.1.zip"
$uri = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"
$expected = "0a6aef473350e9ff4c34302fa0cb1286d409b56c8f228367a4819ebb3ac19e9f"
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
$buildinfo = & $ffexe -buildconf | Out-String
if ($LASTEXITCODE -ne 0) { throw "FFMPEG_BUILD_CONFIG_FAILED" }
if ($buildinfo -match "--enable-gpl" -or $buildinfo -match "--enable-nonfree" -or $buildinfo -match "--enable-libx264" -or $buildinfo -match "--enable-libx265") {
  throw "FFMPEG_GPL_OR_NONFREE_FLAGS_DETECTED"
}
if ($buildinfo -notmatch "--enable-shared") { throw "FFMPEG_IS_NOT_SHARED" }
$buildinfo | Set-Content -Path (Join-Path $destination "FFMPEG_BUILD_CONFIGURATION.txt") -Encoding UTF8
Write-Output "FFMPEG_LGPL_ONLY_SHARED_CONFIGURATION_PASS"
$env:PATH = "$ffBin;$env:PATH"
& python -m pip install --disable-pip-version-check --quiet "Cython>=3,<4" "setuptools>=70,<90" wheel delvewheel "numpy>=1.26,<3"
if ($LASTEXITCODE -ne 0) { throw "PYAV_BUILD_DEPENDENCY_INSTALL_FAILED" }
$pyav = Join-Path $destination "PyAV-v17.1.0-source"
if (!(Test-Path $pyav) ) {
  git clone --depth 1 --branch v17.1.0 https://github.com/PyAV-Org/PyAV.git $pyav
  if ($LASTEXITCODE -ne 0) { throw "PYAV_SOURCE_FETCH_FAILED" }
}
$sourceSha = (git -C $pyav rev-parse HEAD).Trim()
if ($sourceSha -ne "97241598792f4c980c32458187154c7c999d68b3") { throw "PYAV_SOURCE_COMMIT_MISMATCH $sourceSha" }
Write-Output "PYAV_SOURCE_PINNED_PASS $sourceSha"
Push-Location $pyav
try {
  & python setup.py bdist_wheel "--ffmpeg-dir=$ffDir"
  if ($LASTEXITCODE -ne 0) { throw "PYAV_LGPL_BUILD_FAILED" }
}
finally { Pop-Location }
$wheels = @(Get-ChildItem -Path (Join-Path $pyav "dist") -Filter "av-17.1.0-*.whl")
if ($wheels.Count -ne 1) { throw "PYAV_LGPL_WHEEL_COUNT_FAILED $($wheels.Count)" }
$wheelhouse = Join-Path $destination "wheelhouse"
New-Item -ItemType Directory -Path $wheelhouse -Force | Out-Null
& python -m delvewheel repair --add-path $ffBin --wheel-dir $wheelhouse $wheels[0].FullName
if ($LASTEXITCODE -ne 0) { throw "PYAV_LGPL_DELVEWHEEL_FAILED" }
$repaired = @(Get-ChildItem -Path $wheelhouse -Filter "av-17.1.0-*.whl")
if ($repaired.Count -ne 1) { throw "PYAV_LGPL_REPAIRED_WHEEL_COUNT_FAILED $($repaired.Count)" }
& python -m pip uninstall -y av
if ($LASTEXITCODE -ne 0) { throw "OLD_AV_UNINSTALL_FAILED" }
& python -m pip install --no-deps --force-reinstall $repaired[0].FullName
if ($LASTEXITCODE -ne 0) { throw "LGPL_AV_INSTALL_FAILED" }
& python (Join-Path $PSScriptRoot "verify_lgpl_pyav.py") --wheel $repaired[0].FullName --output $destination
if ($LASTEXITCODE -ne 0) { throw "LGPL_AV_VERIFICATION_FAILED" }
Write-Output "PYAV_LGPL_BINARY_GATE_PASS $($repaired[0].FullName)"
