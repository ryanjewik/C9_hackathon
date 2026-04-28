$ErrorActionPreference = "Continue"
$baseDir = "E:\cloud9_hackathon"

# Player arrays are kept as arrays (not joined) so names with spaces
# (e.g. "lovers rock") are passed as proper separate arguments to docker exec.
$vods = @(
    @{ Num=1;  Left=@("skuba","brawk","Ethan","s0m","mada");                          Right=@("Chronicle","Kaajak","Boaster","Alfajer","Crashies") },
    @{ Num=2;  Left=@("Wo0t","MiniBoo","RieNs","benjyfishy","Boo");                   Right=@("d4v41","Jinggg","PatMen","something","f0rsakeN") },
    @{ Num=3;  Left=@("Flashback","BeYN","MaKo","HYUNMIN","free1ng");                 Right=@("paTiTek","kamo","nAts","keiko","trexx") },
    @{ Num=4;  Left=@("Chronicle","Kaajak","Boaster","Alfajer","Crashies");            Right=@("d4v41","Jinggg","PatMen","something","f0rsakeN") },
    @{ Num=5;  Left=@("SiuFatBB","Lysoar","Yuicaw","juicy","Spring");                 Right=@("whzy","rushia","Knight","Nephh","Levius") },
    @{ Num=6;  Left=@("Kamo","kamyk","nAts","paTiTek","Keiko");                       Right=@("bang","zekken","johnqt","Zellsis","N4RRATE") },
    @{ Num=7;  Left=@("bang","Zellsis","zekken","johnqt","N4RRATE");                  Right=@("icy","Derrek","NaturE","yay","supamen") },
    @{ Num=9;  Left=@("purp0","kamo","nAts","wayne","MiniBoo");                       Right=@("Chronicle","Derke","Jamppi","UNFAKE","PROFEK") },
    @{ Num=10; Left=@("Loita","lovers rock","Rose","Lar0k","Crewen");                 Right=@("bipo","GLYPH","starxo","marteen","minny") }
)

foreach ($vod in $vods) {
    $n        = $vod.Num
    $logFile  = "$baseDir\extract_vod${n}_fix22Lv2.log"
    $labelDir = "$baseDir\crops_vod${n}_by_label_fix22Lv2"

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Processing VOD $n" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Clear previous crops and label output in container
    docker exec vod-processor-worker rm -rf /app/outputs/crops /app/outputs/crops_by_label

    # Run extraction — pass Left/Right as arrays so multi-word names stay intact
    Write-Host "Running extraction for VOD $n..."
    docker exec vod-processor-worker python /app/vod_processor/scripts/extract_crops.py `
        --vods $n `
        --left-players $vod.Left `
        --right-players $vod.Right `
        2>&1 | Tee-Object -FilePath $logFile | Select-Object -Last 15

    # Run CNN label export — sorts crops into subfolders by predicted class
    Write-Host "Running CNN classification for VOD $n..."
    docker exec vod-processor-worker python /app/vod_processor/export_crops_by_label.py "crops-vod$n" `
        2>&1 | Tee-Object -Append -FilePath $logFile | Select-Object -Last 10

    # Copy the by-label tree to host
    if (Test-Path $labelDir) { Remove-Item -Recurse -Force $labelDir }
    docker cp vod-processor-worker:/app/outputs/crops_by_label $labelDir

    # Count real crops (exclude *_preprocessed sibling dirs)
    $count = (Get-ChildItem $labelDir -Recurse -Filter "*.png" -ErrorAction SilentlyContinue |
              Where-Object { $_.DirectoryName -notlike "*_preprocessed*" }).Count
    Write-Host "VOD $n complete: $count classified crops -> $labelDir" -ForegroundColor Green
    Write-Host "Log: $logFile" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "  All VODs processed!" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
