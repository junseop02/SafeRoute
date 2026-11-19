param()
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCommand) {
  $nodePath = $nodeCommand.Source
} else {
  $nodePath = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
}
if (-not (Test-Path -LiteralPath $nodePath)) {
  throw 'Node.js를 찾을 수 없습니다. Node.js를 설치한 뒤 다시 실행하세요.'
}
if (-not $env:KAKAO_REST_API_KEY) {
  Write-Host ''
  Write-Host '천안·아산의 48개 읍면동 대표점 사이 경로를 자동 수집합니다.'
  Write-Host '기본 계획: 방향성 2,256쌍 x 3개 공식 우선순위 = 6,768회 요청'
  Write-Host '카카오디벨로퍼스의 REST API 키를 붙여넣고 Enter를 누르세요.'
  Write-Host '키는 화면이나 데이터 파일에 저장되지 않습니다.'
  $secureKey = Read-Host 'REST API 키' -AsSecureString
  $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
  try {
    $env:KAKAO_REST_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
  }
}
& $nodePath (Join-Path $projectDir 'batch-collector.js')
$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
  $resultDirectory = Join-Path $projectDir 'data\processed'
  Start-Process explorer.exe -ArgumentList $resultDirectory
}
exit $exitCode
