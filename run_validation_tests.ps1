$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
& $Python -m unittest -v test_per_camera_calibration test_integrated_math test_api_mock_pipeline test_camera_profiles_api
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python test_classifier_7_scenarios.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
