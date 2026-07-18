$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$BundledPython = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path $BundledPython) {
    $Python = $BundledPython
} else {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

& $Python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
