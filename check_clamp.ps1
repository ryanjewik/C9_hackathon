$lines = Get-Content E:\cloud9_hackathon\extract_log_vod8_strict.txt -Encoding Unicode
$ktrSkip = @{}
$leftClamp = @{}
foreach ($l in $lines) {
  if ($l -match 'crop#(\d+) ktr skip correction') { $ktrSkip[$matches[1]] = $l }
  if ($l -match 'crop#(\d+) left-clamp') { $leftClamp[$matches[1]] = $l }
}
Write-Host "--- Crops with ktr-skip AND left-clamp ---"
foreach ($k in ($ktrSkip.Keys | Sort-Object { [int]$_ })) {
  if ($leftClamp.ContainsKey($k)) {
    Write-Host $ktrSkip[$k]
    Write-Host $leftClamp[$k]
    Write-Host ""
  }
}
Write-Host "--- All left-clamp entries ---"
foreach ($k in ($leftClamp.Keys | Sort-Object { [int]$_ })) {
  Write-Host $leftClamp[$k]
}
