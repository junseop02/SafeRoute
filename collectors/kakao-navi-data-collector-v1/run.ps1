param([switch]$NoBrowser)
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
  Write-Host '카카오디벨로퍼스의 REST API 키를 붙여넣고 Enter를 누르세요.'
  Write-Host '입력한 키는 화면에 표시되거나 파일에 저장되지 않습니다.'
  $secureKey = Read-Host 'REST API 키' -AsSecureString
  $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
  try {
    $env:KAKAO_REST_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
  }
}
if ($NoBrowser) { $env:SAFEROUTE_OPEN_BROWSER = '0' }
& $nodePath (Join-Path $projectDir 'server.js')
