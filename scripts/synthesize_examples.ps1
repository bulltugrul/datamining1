param([Parameter(Mandatory=$true)][string]$ManifestPath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesisStream,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.IInputStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
$manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$synthesizer = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::new()
try {
    $voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -eq 'tr-TR' } | Select-Object -First 1
    if ($null -eq $voice) { throw 'Turkish speech synthesis voice is not installed.' }
    $synthesizer.Voice = $voice
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    foreach ($sample in $manifest.samples) {
        if ($sample.modality -ne 'audio') { continue }
        $operation = $synthesizer.SynthesizeTextToStreamAsync($sample.expected_text)
        $task = $method.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]).Invoke($null, @($operation))
        $task.Wait()
        $speech = $task.Result
        $inputStream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead([Windows.Storage.Streams.IInputStream]$speech)
        $outputStream = [System.IO.File]::Create($sample.absolute_path)
        try { $inputStream.CopyTo($outputStream) }
        finally { $outputStream.Dispose(); $inputStream.Dispose(); $speech.Dispose() }
    }
} finally { $synthesizer.Dispose() }
