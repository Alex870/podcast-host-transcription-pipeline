Add-Type -AssemblyName System.Windows.Forms

$CommonScript = Join-Path $PSScriptRoot "PodcastTranscribeLauncher.Common.ps1"
. $CommonScript

$LauncherContext = Get-PodcastTranscribeLauncherContext -ScriptRoot $PSScriptRoot
$ProjectRoot = $LauncherContext.ProjectRoot

$script:Results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param(
        [string]$Status,
        [string]$Label,
        [string]$Detail
    )

    $script:Results.Add([pscustomobject]@{
        Status = $Status
        Label  = $Label
        Detail = $Detail
    })
}

function Write-ResultSummary {
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "Migration Checklist"
    Write-Host ("=" * 78)

    foreach ($result in $script:Results) {
        switch ($result.Status) {
            "PASS" { $color = "Green" }
            "WARN" { $color = "Yellow" }
            "FAIL" { $color = "Red" }
            default { $color = "Gray" }
        }

        Write-Host -NoNewline ("[{0}]" -f $result.Status) -ForegroundColor $color
        Write-Host (" {0}: {1}" -f $result.Label, $result.Detail)
    }
}

function Select-LegacyFolder {
    param(
        [string]$InitialFolder
    )

    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{
        RootFolder  = "MyComputer"
        Description = "Select the legacy podcast transcription directory to migrate from."
    }

    if ($InitialFolder -and (Test-Path -LiteralPath $InitialFolder)) {
        $dialog.SelectedPath = $InitialFolder
    }

    $selection = $dialog.ShowDialog()
    if ($selection -ne [System.Windows.Forms.DialogResult]::OK) {
        return $null
    }

    return $dialog.SelectedPath
}

function Backup-IfExists {
    param(
        [string]$TargetPath
    )

    if (-not (Test-Path -LiteralPath $TargetPath)) {
        return $null
    }

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = "$TargetPath.migration-backup-$timestamp"
    Move-Item -LiteralPath $TargetPath -Destination $backupPath -Force
    return $backupPath
}

function Copy-FileWithBackup {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        Add-Result "WARN" $Label ("Not found in legacy directory: {0}" -f $SourcePath)
        return
    }

    $targetDir = Split-Path -Parent $TargetPath
    if ($targetDir) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }

    $backupPath = Backup-IfExists -TargetPath $TargetPath
    Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force

    if ($backupPath) {
        Add-Result "PASS" $Label ("Copied to {0}; previous file backed up to {1}" -f $TargetPath, $backupPath)
    } else {
        Add-Result "PASS" $Label ("Copied to {0}" -f $TargetPath)
    }
}

function Copy-DirectoryMerge {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
        Add-Result "WARN" $Label ("Not found in legacy directory: {0}" -f $SourcePath)
        return
    }

    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
    $items = Get-ChildItem -LiteralPath $SourcePath -Force
    if ($items.Count -eq 0) {
        Add-Result "WARN" $Label ("Directory exists but is empty: {0}" -f $SourcePath)
        return
    }

    foreach ($item in $items) {
        Copy-Item -LiteralPath $item.FullName -Destination $TargetPath -Recurse -Force
    }

    Add-Result "PASS" $Label ("Merged {0} item(s) from {1} into {2}" -f $items.Count, $SourcePath, $TargetPath)
}

function Copy-DirectoryWithProgress {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
        Add-Result "WARN" $Label ("Not found in legacy directory: {0}" -f $SourcePath)
        return
    }

    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null

    $sourceRoot = [System.IO.Path]::GetFullPath($SourcePath).TrimEnd('\', '/')
    $files = @(Get-ChildItem -LiteralPath $SourcePath -Recurse -File -Force)
    $directories = @(Get-ChildItem -LiteralPath $SourcePath -Recurse -Directory -Force)

    foreach ($directory in $directories) {
        $relativeDirectory = $directory.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $targetDirectory = Join-Path $TargetPath $relativeDirectory
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
    }

    if ($files.Count -eq 0) {
        Add-Result "WARN" $Label ("Directory exists but contains no files: {0}" -f $SourcePath)
        return
    }

    $activity = "{0}: copying files" -f $Label
    for ($index = 0; $index -lt $files.Count; $index++) {
        $file = $files[$index]
        $relativeFile = $file.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $targetFile = Join-Path $TargetPath $relativeFile
        $targetDirectory = Split-Path -Parent $targetFile
        if ($targetDirectory) {
            New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
        }

        $percentComplete = [int](($index / $files.Count) * 100)
        Write-Progress -Id 1 -Activity $activity -Status ("{0} of {1}: {2}" -f ($index + 1), $files.Count, $relativeFile) -PercentComplete $percentComplete
        Copy-Item -LiteralPath $file.FullName -Destination $targetFile -Force
    }

    Write-Progress -Id 1 -Activity $activity -Completed
    Add-Result "PASS" $Label ("Copied {0} file(s) from {1} into {2}" -f $files.Count, $SourcePath, $TargetPath)
}

function Resolve-LegacyCandidate {
    param(
        [string]$LegacyRoot,
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        $resolved = if ([System.IO.Path]::IsPathRooted($candidate)) {
            $candidate
        } else {
            Join-Path $LegacyRoot $candidate
        }

        if (Test-Path -LiteralPath $resolved) {
            return $resolved
        }
    }

    return $null
}

function Test-PathContainedBy {
    param(
        [string]$Path,
        [string]$Container
    )

    if ([string]::IsNullOrWhiteSpace($Path) -or [string]::IsNullOrWhiteSpace($Container)) {
        return $false
    }

    try {
        $resolvedPath = [System.IO.Path]::GetFullPath($Path)
        $resolvedContainer = [System.IO.Path]::GetFullPath($Container)
    } catch {
        return $false
    }

    $resolvedContainer = $resolvedContainer.TrimEnd('\', '/')
    return $resolvedPath -eq $resolvedContainer -or $resolvedPath.StartsWith("$resolvedContainer\", [System.StringComparison]::OrdinalIgnoreCase)
}

function Convert-ToProjectRelativePath {
    param(
        [string]$Path,
        [string]$BasePath
    )

    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedBase = [System.IO.Path]::GetFullPath($BasePath)
    $uriPath = New-Object System.Uri(($resolvedPath.TrimEnd('\') + '\'))
    $uriBase = New-Object System.Uri(($resolvedBase.TrimEnd('\') + '\'))
    $relative = $uriBase.MakeRelativeUri($uriPath).ToString().TrimEnd('/')
    return [System.Uri]::UnescapeDataString($relative).Replace('/', '\')
}

function Resolve-LegacySourceDirectory {
    param(
        [psobject]$LegacyConfig,
        [string]$LegacyRoot,
        [string]$CurrentProjectRoot
    )

    if ($null -eq $LegacyConfig -or -not $LegacyConfig.default_source_dir) {
        return $null
    }

    $configuredSource = [string]$LegacyConfig.default_source_dir
    $directCandidate = if ([System.IO.Path]::IsPathRooted($configuredSource)) {
        $configuredSource
    } else {
        Join-Path $LegacyRoot $configuredSource
    }

    if ((Test-Path -LiteralPath $directCandidate -PathType Container) -and (Test-PathContainedBy -Path $directCandidate -Container $LegacyRoot)) {
        return [pscustomobject]@{
            SourcePath    = $directCandidate
            RelativePath  = Convert-ToProjectRelativePath -Path $directCandidate -BasePath $LegacyRoot
            Resolution    = "legacy_config_path"
            ConfigValue   = $configuredSource
        }
    }

    if ([System.IO.Path]::IsPathRooted($configuredSource) -and (Test-PathContainedBy -Path $configuredSource -Container $CurrentProjectRoot)) {
        $relativePath = Convert-ToProjectRelativePath -Path $configuredSource -BasePath $CurrentProjectRoot
        $legacyRelativeCandidate = Join-Path $LegacyRoot $relativePath
        if (Test-Path -LiteralPath $legacyRelativeCandidate -PathType Container) {
            return [pscustomobject]@{
                SourcePath    = $legacyRelativeCandidate
                RelativePath  = $relativePath
                Resolution    = "current_repo_relative_fallback"
                ConfigValue   = $configuredSource
            }
        }
    }

    return [pscustomobject]@{
        SourcePath   = $null
        RelativePath = $null
        Resolution   = "not_found"
        ConfigValue  = $configuredSource
    }
}

function Confirm-OverwriteIfNeeded {
    param(
        [System.Collections.Generic.List[object]]$PlannedTargets
    )

    $existingTargets = @($PlannedTargets | ForEach-Object { $_.TargetPath } | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_)
    } | Select-Object -Unique)

    if ($existingTargets.Count -eq 0) {
        return $true
    }

    Write-Host ""
    Write-Host "Warning: the migration will overwrite files or merge into existing target directories." -ForegroundColor Yellow
    Write-Host "Existing targets detected:"
    foreach ($path in $existingTargets | Select-Object -First 8) {
        Write-Host (" - {0}" -f $path)
    }
    if ($existingTargets.Count -gt 8) {
        Write-Host (" - ... and {0} more" -f ($existingTargets.Count - 8))
    }

    $response = (Read-Host "Continue with migration? (Y/N)").Trim().ToUpperInvariant()
    return $response -eq "Y"
}

$legacyRoot = Select-LegacyFolder -InitialFolder $ProjectRoot
if ([string]::IsNullOrWhiteSpace($legacyRoot)) {
    Write-Host "No legacy directory selected. Exiting."
    Read-Host "Press Enter to continue"
    return
}

$resolvedLegacyRoot = (Resolve-Path -LiteralPath $legacyRoot).Path
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

if ($resolvedLegacyRoot -eq $resolvedProjectRoot) {
    Add-Result "WARN" "Legacy directory selection" "Selected directory is the current repository root. No migration was performed."
    Write-ResultSummary
    Read-Host "Press Enter to continue"
    return
}

Add-Result "PASS" "Legacy directory selection" ("Migrating from {0}" -f $resolvedLegacyRoot)
Add-Result "PASS" "Target repository" ("Writing into {0}" -f $resolvedProjectRoot)

$legacyConfigPath = Resolve-LegacyCandidate -LegacyRoot $resolvedLegacyRoot -Candidates @(
    "podcast_transcribe_config.json",
    "examples\podcast_transcribe_config.example.json"
)
$legacyConfig = $null
if ($legacyConfigPath -and (Split-Path -Leaf $legacyConfigPath) -eq "podcast_transcribe_config.json") {
    try {
        $legacyConfig = Get-Content -LiteralPath $legacyConfigPath -Raw | ConvertFrom-Json
    } catch {
        Add-Result "WARN" "Legacy config parsing" ("Could not parse legacy config at {0}: {1}" -f $legacyConfigPath, $_.Exception.Message)
    }
}

$plannedTargets = New-Object 'System.Collections.Generic.List[object]'
foreach ($path in @(
    (Join-Path $resolvedProjectRoot "podcast_transcribe_config.json"),
    (Join-Path $resolvedProjectRoot "preferred_terms.txt"),
    (Join-Path $resolvedProjectRoot "preferred_replacements.json"),
    (Join-Path $resolvedProjectRoot "speaker_reference_samples"),
    (Join-Path $resolvedProjectRoot "pretrained_speaker_model"),
    (Join-Path $resolvedProjectRoot "host_profile.json"),
    (Join-Path $resolvedProjectRoot "_processed_files.json"),
    (Join-Path $resolvedProjectRoot "output")
)) {
    $plannedTargets.Add([pscustomobject]@{ TargetPath = $path }) | Out-Null
}

$legacySourceDir = $null
$targetSourceDir = $null
if ($legacyConfig -and $legacyConfig.default_source_dir) {
    $sourceResolution = Resolve-LegacySourceDirectory -LegacyConfig $legacyConfig -LegacyRoot $resolvedLegacyRoot -CurrentProjectRoot $resolvedProjectRoot
    if ($sourceResolution.SourcePath) {
        $legacySourceDir = $sourceResolution.SourcePath
        $targetSourceDir = Join-Path $resolvedProjectRoot $sourceResolution.RelativePath
        $plannedTargets.Add([pscustomobject]@{ TargetPath = $targetSourceDir }) | Out-Null
        if ($sourceResolution.Resolution -eq "current_repo_relative_fallback") {
            Add-Result "PASS" "Configured source directory resolution" ("Legacy source recovered by matching current-repo-relative path '{0}' inside the legacy directory." -f $sourceResolution.RelativePath.Replace('\', '/'))
        }
    } elseif ($sourceResolution.Resolution -eq "not_found") {
        Add-Result "WARN" "Configured source directory" ("Configured source directory could not be resolved inside the legacy directory: {0}" -f $sourceResolution.ConfigValue)
    } else {
        Add-Result "WARN" "Configured source directory" ("Configured source is outside the legacy directory and will not be copied: {0}" -f $sourceResolution.ConfigValue)
    }
}

if (-not (Confirm-OverwriteIfNeeded -PlannedTargets $plannedTargets)) {
    Add-Result "WARN" "Migration confirmation" "User cancelled migration after overwrite warning."
    Write-ResultSummary
    Read-Host "Press Enter to continue"
    return
}

if ($legacyConfigPath -and (Split-Path -Leaf $legacyConfigPath) -eq "podcast_transcribe_config.json") {
    Copy-FileWithBackup -SourcePath $legacyConfigPath -TargetPath (Join-Path $resolvedProjectRoot "podcast_transcribe_config.json") -Label "Runtime config"
} elseif ($legacyConfigPath) {
    Add-Result "WARN" "Runtime config" ("Found only example config at {0}; skipping because no real runtime config was present." -f $legacyConfigPath)
} else {
    Add-Result "WARN" "Runtime config" "Legacy runtime config was not found."
}

Copy-FileWithBackup `
    -SourcePath (Join-Path $resolvedLegacyRoot "preferred_terms.txt") `
    -TargetPath (Join-Path $resolvedProjectRoot "preferred_terms.txt") `
    -Label "Preferred terms"

Copy-FileWithBackup `
    -SourcePath (Join-Path $resolvedLegacyRoot "preferred_replacements.json") `
    -TargetPath (Join-Path $resolvedProjectRoot "preferred_replacements.json") `
    -Label "Preferred replacements"

$legacySpeakerDir = Resolve-LegacyCandidate -LegacyRoot $resolvedLegacyRoot -Candidates @("speaker_reference_samples")
if ($legacySpeakerDir) {
    Copy-DirectoryMerge `
        -SourcePath $legacySpeakerDir `
        -TargetPath (Join-Path $resolvedProjectRoot "speaker_reference_samples") `
        -Label "Speaker reference samples"

    $legacySpeakersJson = Join-Path $legacySpeakerDir "speakers.json"
    if (Test-Path -LiteralPath $legacySpeakersJson -PathType Leaf) {
        Add-Result "PASS" "Speaker reference config" ("Included speakers.json from {0}" -f $legacySpeakerDir)
    } else {
        Add-Result "WARN" "Speaker reference config" ("speaker_reference_samples exists, but speakers.json was not found in {0}" -f $legacySpeakerDir)
    }
} else {
    Add-Result "WARN" "Speaker reference samples" "Legacy speaker_reference_samples directory was not found."
}

$legacyPretrainedDir = Resolve-LegacyCandidate -LegacyRoot $resolvedLegacyRoot -Candidates @(
    "pretrained_speaker_model",
    "scripts\pretrained_speaker_model_diagnostic",
    "pretrained_speaker_model_diagnostic"
)
if ($legacyPretrainedDir) {
    Copy-DirectoryMerge `
        -SourcePath $legacyPretrainedDir `
        -TargetPath (Join-Path $resolvedProjectRoot "pretrained_speaker_model") `
        -Label "Pretrained speaker model"
} else {
    Add-Result "WARN" "Pretrained speaker model" "No legacy pretrained speaker model directory was found."
}

Copy-FileWithBackup `
    -SourcePath (Join-Path $resolvedLegacyRoot "host_profile.json") `
    -TargetPath (Join-Path $resolvedProjectRoot "host_profile.json") `
    -Label "Host profile"

Copy-FileWithBackup `
    -SourcePath (Join-Path $resolvedLegacyRoot "_processed_files.json") `
    -TargetPath (Join-Path $resolvedProjectRoot "_processed_files.json") `
    -Label "Processed-files state"

$legacyOutputDir = Resolve-LegacyCandidate -LegacyRoot $resolvedLegacyRoot -Candidates @("output")
if ($legacyOutputDir) {
    Copy-DirectoryMerge `
        -SourcePath $legacyOutputDir `
        -TargetPath (Join-Path $resolvedProjectRoot "output") `
        -Label "Output directory contents"
} else {
    Add-Result "WARN" "Output directory contents" "Legacy output directory was not found."
}

if ($legacySourceDir -and $targetSourceDir -and (Test-Path -LiteralPath $legacySourceDir -PathType Container)) {
    Copy-DirectoryWithProgress `
        -SourcePath $legacySourceDir `
        -TargetPath $targetSourceDir `
        -Label "Configured source directory"

    $targetConfigPath = Join-Path $resolvedProjectRoot "podcast_transcribe_config.json"
    if (Test-Path -LiteralPath $targetConfigPath -PathType Leaf) {
        try {
            $targetConfig = Get-Content -LiteralPath $targetConfigPath -Raw | ConvertFrom-Json
            $relativeTargetSource = Convert-ToProjectRelativePath -Path $targetSourceDir -BasePath $resolvedProjectRoot
            $portableSource = $relativeTargetSource.Replace('\', '/')
            if ($null -eq $targetConfig.PSObject.Properties["default_source_dir"]) {
                $targetConfig | Add-Member -NotePropertyName "default_source_dir" -NotePropertyValue $portableSource
            } else {
                $targetConfig.default_source_dir = $portableSource
            }
            $targetConfig | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $targetConfigPath -Encoding UTF8
            Add-Result "PASS" "Updated runtime config source directory" ("default_source_dir now points to {0}" -f $portableSource)
        } catch {
            Add-Result "WARN" "Updated runtime config source directory" ("Could not update migrated config source path: {0}" -f $_.Exception.Message)
        }
    }
}

Write-ResultSummary
Read-Host "Press Enter to continue"
