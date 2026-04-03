$shas = @('6aa7cd7','239ade5','92dd65a','d193fd6','f377389','0c75493','2fe9fc5','3773165','8b8ff1a','15ffa29')
foreach ($sha in $shas) {
    Write-Host "Processing $sha"
    $tree = git show -s --format=%T $sha
    $parents = (git show -s --format=%P $sha).Trim()
    $msg = git show -s --format=%B $sha
    $ad = git show -s --format=%aD $sha
    $cd = git show -s --format=%cD $sha
    $env:GIT_AUTHOR_NAME='Ryan Jewik'
    $env:GIT_AUTHOR_EMAIL='jewik@chapman.edu'
    $env:GIT_AUTHOR_DATE=$ad
    $env:GIT_COMMITTER_NAME='Ryan Jewik'
    $env:GIT_COMMITTER_EMAIL='jewik@chapman.edu'
    $env:GIT_COMMITTER_DATE=$cd
    $args = @($tree)
    if ($parents -ne '') { $parents.Split(' ') | ForEach-Object { $args += '-p'; $args += $_ } }
    $args += '-m'
    $args += $msg
    $new = & git commit-tree @args
    Write-Host "Created replacement commit $new for $sha"
    git replace $sha $new
    Remove-Item Env:GIT_AUTHOR_NAME, Env:GIT_AUTHOR_EMAIL, Env:GIT_AUTHOR_DATE, Env:GIT_COMMITTER_NAME, Env:GIT_COMMITTER_EMAIL, Env:GIT_COMMITTER_DATE -ErrorAction SilentlyContinue
}
# Make replacements permanent on a new branch
Write-Host 'Creating branch ryan_desktop_rewritten from ryan_desktop_rebuild'
& git checkout -B ryan_desktop_rewritten ryan_desktop_rebuild
Write-Host 'Running git filter-branch to rewrite refs using replacements (this may take a while)'
& git filter-branch -- --branches ryan_desktop_rewritten
Write-Host 'Filter-branch complete. Verifying for old email now.'
& git log ryan_desktop_rewritten --pretty=format:"%h %an <%ae> %ad" -n 500 | Select-String "ryan@example.com"
Write-Host 'done'
