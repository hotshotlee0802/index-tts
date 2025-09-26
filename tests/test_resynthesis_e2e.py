import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from indextts.resynthesis import IndexTTS2ResynthesisPipeline


def _generate_sine_wave(path: Path, *, frequency: float = 1000.0, sr: int = 24_000, duration: float = 2.0) -> torch.Tensor:
    """Generate and persist a mono sine tone for testing."""
    num_samples = int(sr * duration)
    time = torch.arange(num_samples, dtype=torch.float32) / sr
    waveform = torch.sin(2 * math.pi * frequency * time).unsqueeze(0)
    torchaudio.save(str(path), waveform, sr)
    return waveform


def test_resynthesis_end_to_end(tmp_path: Path) -> None:
    input_path = tmp_path / "sine_input.wav"
    output_path = tmp_path / "sine_resynth.wav"

    original_waveform = _generate_sine_wave(input_path)

    pipeline = IndexTTS2ResynthesisPipeline(
        cfg_path=Path("checkpoints") / "config.yaml",
        model_dir=Path("checkpoints"),
        device="cpu",
        use_fp16=False,
    )

    semantic, mel, waveform_result = pipeline.resynthesise(input_path)
    assert semantic.ids.numel() > 0
    assert mel.mel.numel() > 0

    torchaudio.save(str(output_path), waveform_result.audio, waveform_result.sampling_rate)

    resampled_original = torchaudio.functional.resample(
        original_waveform, 24_000, waveform_result.sampling_rate
    )

    min_length = min(resampled_original.size(-1), waveform_result.audio.size(-1))
    resampled_original = resampled_original[..., :min_length]
    reconstructed = waveform_result.audio[..., :min_length]

    mse = torch.mean((resampled_original - reconstructed) ** 2).item()
    l1 = torch.mean((resampled_original - reconstructed).abs()).item()
    power = torch.mean(resampled_original**2).item()
    snr = float("inf") if mse == 0 else 10 * math.log10(power / (mse + 1e-12))

    print(
        {
            "sampling_rate": waveform_result.sampling_rate,
            "mse": mse,
            "l1": l1,
            "snr_db": snr,
        }
    )

    assert torch.isfinite(torch.tensor([mse, l1, snr])).all()

    assert output_path.exists()
