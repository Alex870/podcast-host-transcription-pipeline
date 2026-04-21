$PythonScript = Join-Path $PSScriptRoot "podcast_transcribe_host.py"
$ConfigPath = Join-Path $PSScriptRoot "podcast_transcribe_config.json"

function Resolve-ConfigPathValue {
    param(
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }

    return Join-Path $PSScriptRoot $Value
}

$Config = $null
if (Test-Path -LiteralPath $ConfigPath) {
    $Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
}

$PreferredTermsFile = Resolve-ConfigPathValue $(if ($Config.preferred_terms_file) { $Config.preferred_terms_file } else { "preferred_terms.txt" })
$ReplacementMapJson = Resolve-ConfigPathValue $(if ($Config.replacement_map_json) { $Config.replacement_map_json } else { "preferred_replacements.json" })
$HostProfileJson = Resolve-ConfigPathValue $(if ($Config.host_profile_json) { $Config.host_profile_json } else { "host_profile.json" })
$KnownSpeakersDir = Resolve-ConfigPathValue $(if ($Config.known_speakers_dir) { $Config.known_speakers_dir } else { "speaker_reference_samples" })

$WhisperModel = if ($Config.model) { $Config.model } else { "large-v3" }
$Language = if ($Config.language) { $Config.language } else { "en" }
$ComputeType = if ($Config.compute_type) { $Config.compute_type } else { "auto" }
$BeamSize = if ($null -ne $Config.beam_size) { [int]$Config.beam_size } else { 5 }
$BatchSize = if ($null -ne $Config.batch_size) { [int]$Config.batch_size } else { 8 }
$AssumeDominantSpeakerIsHost = if ($null -ne $Config.assume_dominant_speaker_is_host) { [bool]$Config.assume_dominant_speaker_is_host } else { $true }
$HostThreshold = if ($null -ne $Config.host_threshold) { [double]$Config.host_threshold } else { 0.45 }
$DefaultSourceFolder = Resolve-ConfigPathValue $(if ($Config.default_source_dir) { $Config.default_source_dir } else { $null })
$ConfiguredHfToken = if ($Config.hf_token) { [string]$Config.hf_token } else { $null }

Write-Host "Folder selection dialog open..."
Add-Type -AssemblyName System.Windows.Forms

$FolderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{
    RootFolder  = "MyComputer"
    Description = "Select a source folder for podcast audio files."
}

if ($DefaultSourceFolder -and (Test-Path -LiteralPath $DefaultSourceFolder)) {
    $FolderBrowser.SelectedPath = $DefaultSourceFolder
}

$null = $FolderBrowser.ShowDialog()

if ([string]::IsNullOrWhiteSpace($FolderBrowser.SelectedPath)) {
    Write-Error "Error: folder not selected. Exiting."
    pause
    exit
}

$OpenFileDialog = New-Object System.Windows.Forms.OpenFileDialog -Property @{
    Title = "Optional: choose a clean host voice sample"
    Filter = "Audio Files|*.mp3;*.wav;*.m4a;*.flac;*.ogg|All Files|*.*"
    Multiselect = $false
}

$HostReference = $null
if ($OpenFileDialog.ShowDialog() -eq "OK" -and -not [string]::IsNullOrWhiteSpace($OpenFileDialog.FileName)) {
    $HostReference = $OpenFileDialog.FileName
}

if (-not (Test-Path -LiteralPath $PythonScript)) {
    Write-Error "Python helper script not found: $PythonScript"
    pause
    exit
}

if (-not $env:HF_TOKEN -and -not [string]::IsNullOrWhiteSpace($ConfiguredHfToken)) {
    $env:HF_TOKEN = $ConfiguredHfToken
}

if (-not $env:HF_TOKEN) {
    Write-Host ""
    Write-Host "HF_TOKEN is not set."
    Write-Host "Set hf_token in podcast_transcribe_config.json or set HF_TOKEN in the environment before running this script."
    Write-Host "It is required for pyannote speaker diarization and speaker attribution."
    pause
    exit
}

conda activate whisper
cls

$SourceFolder = $FolderBrowser.SelectedPath
$startTime = Get-Date

Write-Host "Processing Folder: $SourceFolder"
if ($HostReference) {
    Write-Host "Host reference sample: $HostReference"
} else {
    Write-Host "Host reference sample: none selected"
}

$args = @(
    $PythonScript
    "--input-dir", $SourceFolder
    "--output-dir", $SourceFolder
    "--model", $WhisperModel
    "--language", $Language
    "--compute-type", $ComputeType
    "--beam-size", "$BeamSize"
    "--batch-size", "$BatchSize"
    "--preferred-terms-file", $PreferredTermsFile
    "--replacement-map-json", $ReplacementMapJson
    "--host-profile-json", $HostProfileJson
    "--host-threshold", "$HostThreshold"
)

if ($AssumeDominantSpeakerIsHost) {
    $args += "--assume-dominant-speaker-is-host"
}

if ($HostReference) {
    $args += @("--host-reference", $HostReference)
}

if (Test-Path -LiteralPath (Join-Path $KnownSpeakersDir "speakers.json")) {
    $args += @("--known-speakers-dir", $KnownSpeakersDir)
}

& python @args
$pythonExitCode = $LASTEXITCODE

if ($pythonExitCode -ne 0) {
    Write-Host ""
    Write-Host "The transcription pipeline exited with an error."
    Write-Host "If the error mentions authentication, access denied, unauthorized, or pyannote model loading,"
    Write-Host "your Hugging Face token is likely missing, invalid, or has not accepted the pyannote model access terms."
    Write-Host "Update hf_token in podcast_transcribe_config.json or HF_TOKEN in the environment and confirm access to"
    Write-Host "pyannote/speaker-diarization-community-1 on Hugging Face."
}

$elapsedTime = (Get-Date) - $startTime
Write-Host ("Total Duration: {0} hr {1} min {2} sec" -f $elapsedTime.Hours, $elapsedTime.Minutes, $elapsedTime.Seconds)

pause
