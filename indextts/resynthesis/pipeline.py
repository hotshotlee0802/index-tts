"""IndexTTS2 audio re-synthesis pipeline.

This module implements the three-stage workflow outlined in the project
specification. Given an input waveform, the pipeline performs the
following steps:

1. Use MaskGCT's semantic codec to extract discrete semantic ids from the
   waveform.
2. Run the IndexTTS2 diffusion model to convert the semantic sequence
   back to a mel spectrogram that mirrors the reference audio.
3. Decode the mel spectrogram with the IndexTTS2 BigVGAN vocoder to
   recover the waveform.

Each stage is implemented to match the behaviour of the main
``indextts.infer_v2.IndexTTS2`` entry point so that the configuration,
sampling parameters, and model checkpoints remain consistent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from transformers import SeamlessM4TFeatureExtractor

from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model


@dataclass
class SemanticEncodingResult:
    """Container for semantic codec outputs."""

    ids: torch.LongTensor
    """Quantised semantic token ids produced by the codec."""

    features: Optional[torch.Tensor] = None
    """Optional intermediate representations (e.g. embeddings or logits)."""

    metadata: Optional[Dict[str, Any]] = None
    """Auxiliary information emitted during encoding."""


@dataclass
class MelSynthesisResult:
    """Container for mel diffusion outputs."""

    mel: torch.Tensor
    """Generated mel spectrogram aligned with the semantic ids."""

    metadata: Optional[Dict[str, Any]] = None
    """Auxiliary information emitted during mel generation."""


@dataclass
class WaveformResult:
    """Container for reconstructed waveform segments."""

    audio: torch.Tensor
    """A batch of waveform tensors in the target sampling rate."""

    sampling_rate: int
    """Sampling rate that matches the vocoder configuration."""

    metadata: Optional[Dict[str, Any]] = None
    """Auxiliary information emitted during vocoder inference."""


class IndexTTS2ResynthesisPipeline:
    """High-level orchestrator for IndexTTS2 audio resynthesis.

    Parameters
    ----------
    cfg_path:
        Path to the IndexTTS2 configuration file. This is the same YAML
        file consumed by :class:`indextts.infer_v2.IndexTTS2`.
    model_dir:
        Directory that stores the checkpoints required by IndexTTS2.
    device:
        Target torch device. If not provided the pipeline determines an
        appropriate device lazily when models are instantiated.
    use_fp16:
        Flag indicating whether half precision should be requested when
        loading the diffusion and vocoder models.
    load_on_init:
        When ``True`` the pipeline performs the model loading procedures
        immediately. The actual loading logic is intentionally deferred to
        dedicated helper methods so that it can be filled in afterwards.
    """

    def __init__(
        self,
        cfg_path: str | Path,
        model_dir: str | Path,
        *,
        device: Optional[str] = None,
        use_fp16: bool = False,
        load_on_init: bool = False,
    ) -> None:
        self.cfg_path = Path(cfg_path)
        self.model_dir = Path(model_dir)
        self.requested_device = device
        self.use_fp16 = use_fp16

        self._cfg: Optional[OmegaConf] = None
        self._device: Optional[torch.device] = None
        # Placeholders for the three major components of the pipeline.
        self.semantic_codec: Optional[torch.nn.Module] = None
        self.diffusion_model: Optional[torch.nn.Module] = None
        self.vocoder: Optional[torch.nn.Module] = None

        # Auxiliary components required by the individual stages.
        self._feature_extractor: Optional[SeamlessM4TFeatureExtractor] = None
        self._semantic_model: Optional[torch.nn.Module] = None
        self._semantic_mean: Optional[torch.Tensor] = None
        self._semantic_std: Optional[torch.Tensor] = None
        self._campplus_model: Optional[torch.nn.Module] = None
        self._mel_fn: Optional[Any] = None

        if load_on_init:
            self.load_components()

    # ------------------------------------------------------------------
    # Device and configuration helpers
    # ------------------------------------------------------------------
    @property
    def cfg(self) -> OmegaConf:
        """Lazy accessor for the IndexTTS2 configuration."""

        if self._cfg is None:
            self._cfg = OmegaConf.load(self.cfg_path)
        return self._cfg

    @property
    def device(self) -> torch.device:
        """Resolve the torch device used across all components."""

        if self._device is None:
            self._device = self._autodetect_device()
        return self._device

    def _autodetect_device(self) -> torch.device:
        """Infer an appropriate torch device for inference.

        The detection order mirrors the logic implemented in
        :class:`indextts.infer_v2.IndexTTS2` so that downstream behaviour is
        consistent with the main inference entry point.
        """

        if self.requested_device:
            return torch.device(self.requested_device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            return torch.device("xpu")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Model loading hooks
    # ------------------------------------------------------------------
    def load_components(self) -> None:
        """Initialise the semantic codec, diffusion model, and vocoder.

        The method delegates to :meth:`_load_semantic_codec`,
        :meth:`_load_diffusion_model`, and :meth:`_load_vocoder` so that each
        piece can be implemented independently.
        """

        self.semantic_codec = self._load_semantic_codec()
        self.diffusion_model = self._load_diffusion_model()
        self.vocoder = self._load_vocoder()

    def _load_semantic_codec(self) -> torch.nn.Module:
        """Instantiate the MaskGCT semantic codec used in step 1.

        Returns
        -------
        torch.nn.Module
            A module capable of converting waveforms to semantic ids. The
            concrete implementation will mirror ``build_semantic_codec`` from
            :mod:`indextts.utils.maskgct_utils`.
        """

        codec = build_semantic_codec(self.cfg.semantic_codec)
        checkpoint_path = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        load_model(codec, checkpoint_path)

        codec = codec.to(self.device)
        codec.eval()

        # Load the supporting semantic encoder used to produce features for
        # the codec. The helper mirrors ``IndexTTS2``.
        semantic_model, semantic_mean, semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat)
        )
        self._semantic_model = semantic_model.to(self.device)
        self._semantic_model.eval()
        self._semantic_mean = semantic_mean.to(self.device)
        self._semantic_std = semantic_std.to(self.device)

        self._feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        return codec

    def _load_diffusion_model(self) -> torch.nn.Module:
        """Instantiate the IndexTTS2 diffusion (s2mel) model for step 2."""
        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )

        s2mel = s2mel.to(self.device)
        s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        s2mel.eval()

        # Prepare the mel-spectrogram helper so that we can reuse it during
        # semantic encoding.
        spect_params = self.cfg.s2mel["preprocess_params"]["spect_params"]
        mel_kwargs = {
            "n_fft": spect_params["n_fft"],
            "win_size": spect_params["win_length"],
            "hop_size": spect_params["hop_length"],
            "num_mels": spect_params["n_mels"],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": spect_params.get("fmin", 0),
            "fmax": None if spect_params.get("fmax", "None") == "None" else 8000,
            "center": False,
        }
        self._mel_fn = lambda x: mel_spectrogram(x, **mel_kwargs)

        # Load CAMPPlus for global style extraction.
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model = campplus_model.to(self.device)
        campplus_model.eval()
        self._campplus_model = campplus_model

        return s2mel

    def _load_vocoder(self) -> torch.nn.Module:
        """Instantiate the IndexTTS2 vocoder (BigVGAN) for step 3."""
        bigvgan_name = self.cfg.vocoder.name
        use_cuda_kernel = self.device.type == "cuda"
        try:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                bigvgan_name, use_cuda_kernel=use_cuda_kernel
            )
        except Exception:
            vocoder = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)

        vocoder = vocoder.to(self.device)
        vocoder.remove_weight_norm()
        vocoder.eval()
        return vocoder

    # ------------------------------------------------------------------
    # Stage specific APIs
    # ------------------------------------------------------------------
    def encode_semantics(self, audio_path: str | Path) -> SemanticEncodingResult:
        """Convert an input waveform into semantic ids using MaskGCT.

        Parameters
        ----------
        audio_path:
            Path to the ``.wav`` file that should be re-synthesised.

        Returns
        -------
        SemanticEncodingResult
            Semantic ids together with cached tensors required by the
            downstream diffusion and vocoder stages.
        """

        if self.semantic_codec is None:
            self.semantic_codec = self._load_semantic_codec()
        assert self.semantic_codec is not None
        assert self._feature_extractor is not None
        assert self._semantic_model is not None
        assert self._semantic_mean is not None
        assert self._semantic_std is not None

        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.to(torch.float32)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = int(self.cfg.s2mel["preprocess_params"]["sr"])
        audio_16k = self._resample_waveform(waveform, sr, 16000)
        audio_target = self._resample_waveform(waveform, sr, target_sr)

        inputs = self._feature_extractor(
            audio_16k.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            semantic_outputs = self._semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = semantic_outputs.hidden_states[17]
            feat = (feat - self._semantic_mean) / self._semantic_std
            semantic_ids, reconstructed_feat = self.semantic_codec.quantize(feat)

        metadata = {
            "audio_16k": audio_16k.cpu(),
            "audio_target": audio_target.cpu(),
            "target_sr": target_sr,
            "original_sr": sr,
            "attention_mask": attention_mask.cpu(),
        }

        return SemanticEncodingResult(
            ids=semantic_ids.detach().cpu(),
            features=reconstructed_feat.detach().cpu(),
            metadata=metadata,
        )

    def semantic_to_mel(
        self,
        encoding: SemanticEncodingResult,
        *,
        prompt_condition: Optional[torch.Tensor] = None,
        style_embedding: Optional[torch.Tensor] = None,
        diffusion_kwargs: Optional[Dict[str, Any]] = None,
    ) -> MelSynthesisResult:
        """Run the IndexTTS2 diffusion model to reconstruct mels.

        Parameters
        ----------
        encoding:
            Output of :meth:`encode_semantics` containing the semantic ids.
        prompt_condition:
            Optional prompt conditioning tensor following the expectations of
            the IndexTTS2 length regulator.
        style_embedding:
            Optional global style embedding derived from CAMPPlus.
        diffusion_kwargs:
            Model-specific overrides that control the sampling process.
        """

        if self.semantic_codec is None:
            self.semantic_codec = self._load_semantic_codec()
        if self.diffusion_model is None:
            self.diffusion_model = self._load_diffusion_model()
        assert self.semantic_codec is not None
        assert self.diffusion_model is not None
        assert self._mel_fn is not None

        diffusion_kwargs = diffusion_kwargs or {}
        diffusion_steps = diffusion_kwargs.get("diffusion_steps", 25)
        inference_cfg_rate = diffusion_kwargs.get("inference_cfg_rate", 0.7)
        temperature = diffusion_kwargs.get("temperature", 1.0)

        device = self.device
        ids = encoding.ids.to(device)

        with torch.no_grad():
            semantic_embeddings = self.semantic_codec.quantizer.vq2emb(ids.unsqueeze(1))
            semantic_embeddings = semantic_embeddings.transpose(1, 2)

            metadata = encoding.metadata or {}
            audio_target = metadata.get("audio_target")
            if audio_target is None:
                raise ValueError("Semantic encoding metadata is missing the resampled audio.")
            audio_target = audio_target.to(device)
            ref_mel = self._mel_fn(audio_target.float())
            ref_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)

            if encoding.features is None:
                raise ValueError("Semantic features are required to build the prompt condition.")
            ref_embeddings = encoding.features.to(device)
            computed_prompt = self.diffusion_model.models["length_regulator"](
                ref_embeddings, ylens=ref_lengths, n_quantizers=3, f0=None
            )[0]

            target_condition = self.diffusion_model.models["length_regulator"](
                semantic_embeddings, ylens=ref_lengths, n_quantizers=3, f0=None
            )[0]

            if prompt_condition is not None:
                prompt_condition = prompt_condition.to(device)
            else:
                prompt_condition = computed_prompt

            if style_embedding is not None:
                style = style_embedding.to(device)
            else:
                style = self._compute_style_embedding(metadata)
                style = style.to(device)

            cat_condition = torch.cat([prompt_condition, target_condition], dim=1)
            condition_lengths = torch.LongTensor([cat_condition.size(1)]).to(device)

            mel = self.diffusion_model.models["cfm"].inference(
                cat_condition,
                condition_lengths,
                ref_mel,
                style,
                None,
                diffusion_steps,
                temperature=temperature,
                inference_cfg_rate=inference_cfg_rate,
            )
            mel = mel[:, :, ref_mel.size(-1) :]

        metadata_out = {
            "ref_mel": ref_mel.detach().cpu(),
            "style": style.detach().cpu(),
            "condition_lengths": condition_lengths.cpu(),
            "diffusion_settings": {
                "diffusion_steps": diffusion_steps,
                "inference_cfg_rate": inference_cfg_rate,
                "temperature": temperature,
            },
        }

        return MelSynthesisResult(mel=mel.detach().cpu(), metadata=metadata_out)

    def mel_to_waveform(
        self,
        mel_result: MelSynthesisResult,
        *,
        chunking: Optional[Dict[str, Any]] = None,
    ) -> WaveformResult:
        """Run the IndexTTS2 vocoder to reconstruct the waveform."""
        if self.vocoder is None:
            self.vocoder = self._load_vocoder()
        assert self.vocoder is not None

        device = self.device
        mel = mel_result.mel.to(device).float()

        with torch.no_grad():
            waveform = self.vocoder(mel).squeeze(1)

        waveform = waveform.detach().cpu()
        sampling_rate = int(self.cfg.s2mel["preprocess_params"]["sr"])
        metadata = {"chunking": chunking}
        return WaveformResult(audio=waveform, sampling_rate=sampling_rate, metadata=metadata)

    # ------------------------------------------------------------------
    # End-to-end helper
    # ------------------------------------------------------------------
    def resynthesise(
        self,
        audio_path: str | Path,
        *,
        diffusion_kwargs: Optional[Dict[str, Any]] = None,
        vocoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SemanticEncodingResult, MelSynthesisResult, WaveformResult]:
        """Run the three-stage audio resynthesis workflow.

        The method ties the stage-specific hooks together while keeping the
        plumbing explicit so that intermediate artefacts are readily
        inspectable during development and testing.
        """

        semantic = self.encode_semantics(audio_path)
        mel = self.semantic_to_mel(semantic, diffusion_kwargs=diffusion_kwargs)
        waveform = self.mel_to_waveform(mel, chunking=vocoder_kwargs)
        return semantic, mel, waveform

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resample_waveform(self, waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
        if sr == target_sr:
            return waveform.clone()
        return torchaudio.functional.resample(waveform, sr, target_sr)

    def _compute_style_embedding(self, metadata: Dict[str, Any]) -> torch.Tensor:
        if self._campplus_model is None:
            self.diffusion_model = self._load_diffusion_model()
        assert self._campplus_model is not None

        audio_16k = metadata.get("audio_16k")
        if audio_16k is None:
            raise ValueError("Semantic encoding metadata is missing the 16 kHz audio.")

        device = self.device
        audio_16k = audio_16k.to(device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self._campplus_model(feat.unsqueeze(0))
        return style


__all__ = [
    "IndexTTS2ResynthesisPipeline",
    "SemanticEncodingResult",
    "MelSynthesisResult",
    "WaveformResult",
]
