param(
    [string]$GatewayUrl = "http://localhost:18080",
    [int]$RateAttempts = 120
)

Write-Host "Using GATEWAY=$GatewayUrl (PowerShell runner)"

$ts = Get-Date -Format yyyyMMddHHmmss
$USERNAME = "testuser$ts"
$EMAIL = "test.user.$ts@example.com"
$PASSWORD = "P@ssw0rd123!"
$TEAM_NAME = "test-team-$ts"

function Post-Json($url, $body, $headers=@{}){
    $json = $body | ConvertTo-Json -Depth 10
    return Invoke-RestMethod -Uri $url -Method Post -Body $json -ContentType 'application/json' -Headers $headers -ErrorAction Stop
}

try{
    Write-Host "`n==> 1) Registering user"
    $regBody = @{ username = $USERNAME; email = $EMAIL; password = $PASSWORD }
    $regResp = Post-Json "$GatewayUrl/auth/register" $regBody
    Write-Host "Registered user OK"
} catch {
    Write-Error "Register failed: $($_.Exception.Response.StatusCode) - $($_.Exception.Message)"
    throw
}

try{
    Write-Host "`n==> 2) Logging in to get JWT"
    $loginBody = @{ username = $USERNAME; password = $PASSWORD }
    $loginResp = Post-Json "$GatewayUrl/auth/login" $loginBody
    if ($null -ne $loginResp.access_token) { $JWT = $loginResp.access_token } elseif ($null -ne $loginResp.token) { $JWT = $loginResp.token } else { $JWT = $null }
    if (-not $JWT) { throw "Failed to extract JWT from login response" }
    Write-Host "Obtained JWT"
} catch {
    Write-Error "Login failed: $($_.Exception.Message)"
    throw
}

try{
    Write-Host "`n==> 3) Creating a team"
    $teamBody = @{ name = $TEAM_NAME }
    $headers = @{ Authorization = "Bearer $JWT" }
    $teamResp = Post-Json "$GatewayUrl/teamadmin/create" $teamBody $headers
    if ($null -ne $teamResp.team_id) { $TEAM_ID = $teamResp.team_id } elseif ($null -ne $teamResp.id) { $TEAM_ID = $teamResp.id } else { $TEAM_ID = $null }
    if (-not $TEAM_ID) { throw "Failed to extract team id" }
    Write-Host "Created team $TEAM_ID"
} catch {
    Write-Error "Create team failed: $($_.Exception.Message)"
    throw
}

try{
    Write-Host "`n==> 4) Creating API key"
    $apiBody = @{ name = "ci-test-key" }
    $apiResp = Post-Json "$GatewayUrl/teamadmin/$TEAM_ID/apikeys" $apiBody $headers
    if ($null -ne $apiResp.key) { $API_KEY = $apiResp.key } elseif ($null -ne $apiResp.token) { $API_KEY = $apiResp.token } else { $API_KEY = $null }
    if ($API_KEY) { Write-Host "API key created" } else { Write-Host "API key not returned (continuing)" }
} catch {
    Write-Warning "API key creation failed (continuing): $($_.Exception.Message)"
}

# Exchange raw API key for a short-lived API key JWT (required for /api/** routes)
$APIKEY_JWT = $null
if ($API_KEY) {
    try {
        $tokenResp = Post-Json "$GatewayUrl/auth/token" @{ key = $API_KEY }
        $APIKEY_JWT = if ($null -ne $tokenResp.access_token) { $tokenResp.access_token } else { $null }
        if ($APIKEY_JWT) { Write-Host "API key JWT obtained" } else { Write-Warning "Failed to obtain API key JWT" }
    } catch {
        Write-Warning "API key token exchange failed: $($_.Exception.Message)"
    }
}

Write-Host "`n==> 5) Calling data service via gateway with API key JWT"
$DATA_TEST_ENDPOINT = "$GatewayUrl/api/teams"
if ($APIKEY_JWT) {
    try{
        $sw = [diagnostics.stopwatch]::StartNew()
        $resp = Invoke-RestMethod -Uri $DATA_TEST_ENDPOINT -Headers @{ Authorization = "Bearer $APIKEY_JWT" } -ErrorAction Stop
        $sw.Stop()
        Write-Host "Gateway -> Data service OK (time ${($sw.Elapsed.TotalSeconds)}s)"
    } catch {
        Write-Warning "Gateway call failed: $($_.Exception.Message)"
    }
} else { Write-Warning "No API key JWT available; skipping data service test" }

if ($API_KEY) {
    Write-Host "`n==> 6) Calling data service via gateway with API key (token exchange)"
    if ($APIKEY_JWT) {
        try{
            $sw = [diagnostics.stopwatch]::StartNew()
            $resp2 = Invoke-RestMethod -Uri $DATA_TEST_ENDPOINT -Headers @{ Authorization = "Bearer $APIKEY_JWT" } -ErrorAction Stop
            $sw.Stop()
            Write-Host "API key auth OK (time ${($sw.Elapsed.TotalSeconds)}s)"
        } catch {
            Write-Warning "API key call failed: $($_.Exception.Message)"
        }
    } else { Write-Warning "No API key JWT; skipping" }
} else { Write-Host "No API key available; skipping API-key auth test" }

Write-Host "`n==> 7) Caching test (expect second request to be served from cache)"
$CACHE_ENDPOINT = "$GatewayUrl/api/teams"
if ($APIKEY_JWT) {
    try{
        $t1 = [datetime]::UtcNow
        $b1 = Invoke-RestMethod -Uri $CACHE_ENDPOINT -Headers @{ Authorization = "Bearer $APIKEY_JWT" } -ErrorAction Stop
        $t2 = [datetime]::UtcNow
        Start-Sleep -Milliseconds 200
        $t3 = [datetime]::UtcNow
        $b2 = Invoke-RestMethod -Uri $CACHE_ENDPOINT -Headers @{ Authorization = "Bearer $APIKEY_JWT" } -ErrorAction Stop
        $t4 = [datetime]::UtcNow
        $time1 = ($t2 - $t1).TotalSeconds
        $time2 = ($t4 - $t3).TotalSeconds
        Write-Host "First call time=$time1, Second call time=$time2"
        if ((ConvertTo-Json $b1) -eq (ConvertTo-Json $b2)) { Write-Host "Bodies match" } else { Write-Warning "Bodies differ" }
        if ($time2 -le $time1) { Write-Host "Second request equal-or-faster (cache likely working)" } else { Write-Warning "Second request slower (cache may not be active)" }
    } catch {
        Write-Warning "Caching test failed: $($_.Exception.Message)"
    }
} else { Write-Warning "No API key JWT; skipping caching test" }

Write-Host "`n==> 8) Sliding-window rate-limiter test"
$RATE_ENDPOINT = "$GatewayUrl/api/players"
$limitHit = $false
if ($APIKEY_JWT) {
    for ($i=1; $i -le $RateAttempts; $i++){
        try{
            $r = Invoke-RestMethod -Uri $RATE_ENDPOINT -Headers @{ Authorization = "Bearer $APIKEY_JWT" } -ErrorAction Stop
        } catch {
            if ($_.Exception.Response -and $_.Exception.Response.StatusCode -eq 429) {
                Write-Host "Received 429 on attempt $i"
                $limitHit = $true; break
            }
        }
        Start-Sleep -Milliseconds 50
    }
}
if ($limitHit) { Write-Host "Rate limiter triggered as expected" } else { Write-Warning "Rate limiter did not trigger" }

Write-Host "`nIntegration tests complete."
