"""
GENESIS-DSP — Adım 06
Yeniden kullanılabilir birleşik bozunum pipeline'ı.

Bu program:
1. Adım 02 temiz SignalRecord paketini yükler.
2. Bozunumları tek bir yapılandırılmış pipeline içinde sırayla uygular:
   CFO -> faz -> multipath -> fractional delay -> IQ imbalance
   -> DC offset -> clipping -> AWGN
3. Her ara aşamayı saklar.
4. Aynı seed ile aynı sonucu verdiğini doğrular.
5. Pipeline konfigürasyonu, veri paketi, metadata ve grafik üretir.

Çalıştırma:
    python 06_impairment_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP02_DIRECTORY = BASE_DIRECTORY / "outputs" / "step02"
STEP06_DIRECTORY = BASE_DIRECTORY / "outputs" / "step06"

INPUT_PACKAGE = STEP02_DIRECTORY / "signal_record.npz"
INPUT_METADATA = STEP02_DIRECTORY / "signal_record_metadata.json"

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImpairmentConfig:
    """Bir birleşik bozunum senaryosunun bütün parametreleri."""

    random_seed: int = 20260722

    cfo_hz: float = 7_500.0
    phase_offset_degrees: float = 32.0

    path_delays_samples: tuple[int, ...] = (0, 3, 8)
    path_magnitudes: tuple[float, ...] = (1.0, 0.48, 0.23)
    path_phases_degrees: tuple[float, ...] = (0.0, 42.0, -71.0)

    fractional_delay_samples: float = 0.37
    fractional_delay_filter_taps: int = 41

    iq_gain_imbalance_db: float = 1.20
    iq_phase_imbalance_degrees: float = 5.0

    dc_offset_i_rms_ratio: float = 0.12
    dc_offset_q_rms_ratio: float = -0.08

    clipping_level_rms_multiplier: float = 1.25
    target_snr_db: float = 12.0

    def validate(self, sample_rate_hz: float) -> None:
        if sample_rate_hz <= 0.0 or not np.isfinite(sample_rate_hz):
            raise ValueError("sample_rate_hz pozitif ve sonlu olmalıdır.")

        if not np.isfinite(self.cfo_hz):
            raise ValueError("cfo_hz sonlu olmalıdır.")

        if abs(self.cfo_hz) >= sample_rate_hz / 2.0:
            raise ValueError("CFO büyüklüğü Nyquist frekansından küçük olmalıdır.")

        if not np.isfinite(self.phase_offset_degrees):
            raise ValueError("phase_offset_degrees sonlu olmalıdır.")

        if not (
            len(self.path_delays_samples)
            == len(self.path_magnitudes)
            == len(self.path_phases_degrees)
        ):
            raise ValueError(
                "Multipath gecikme, büyüklük ve faz listeleri aynı uzunlukta olmalıdır."
            )

        if len(self.path_delays_samples) == 0:
            raise ValueError("En az bir multipath yolu bulunmalıdır.")

        if any(delay < 0 for delay in self.path_delays_samples):
            raise ValueError("Multipath gecikmeleri negatif olamaz.")

        if len(set(self.path_delays_samples)) != len(self.path_delays_samples):
            raise ValueError("Multipath gecikmeleri benzersiz olmalıdır.")

        if any(magnitude < 0.0 for magnitude in self.path_magnitudes):
            raise ValueError("Multipath büyüklükleri negatif olamaz.")

        if not -0.5 < self.fractional_delay_samples < 0.5:
            raise ValueError(
                "fractional_delay_samples -0.5 ile +0.5 arasında olmalıdır."
            )

        if (
            self.fractional_delay_filter_taps < 5
            or self.fractional_delay_filter_taps % 2 == 0
        ):
            raise ValueError(
                "fractional_delay_filter_taps en az 5 ve tek sayı olmalıdır."
            )

        if self.clipping_level_rms_multiplier <= 0.0:
            raise ValueError("Clipping çarpanı pozitif olmalıdır.")

        scalar_values = (
            self.iq_gain_imbalance_db,
            self.iq_phase_imbalance_degrees,
            self.dc_offset_i_rms_ratio,
            self.dc_offset_q_rms_ratio,
            self.target_snr_db,
        )

        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("Bütün bozunum parametreleri sonlu olmalıdır.")


@dataclass(frozen=True)
class CleanSignalPackage:
    samples: ComplexArray
    bits: NDArray[np.int8]
    symbols: ComplexArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class PipelineResult:
    clean_samples: ComplexArray
    carrier_impaired_samples: ComplexArray
    multipath_samples: ComplexArray
    timing_impaired_samples: ComplexArray
    iq_impaired_samples: ComplexArray
    dc_offset_samples: ComplexArray
    clipped_samples: ComplexArray
    received_samples: ComplexArray
    noise_samples: ComplexArray
    channel_taps: ComplexArray
    fractional_delay_taps: FloatArray
    phase_trajectory_radians: FloatArray
    dc_offset: complex
    clipping_level: float
    clipped_fraction: float
    achieved_snr_db: float
    iq_image_rejection_db: float


def array_sha256(array: NDArray[Any]) -> str:
    """NumPy dizisinin byte içeriğinden SHA-256 özeti üretir."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def load_clean_signal() -> CleanSignalPackage:
    """Adım 02 standart temiz sinyal paketini yükler."""

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 02 veri paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python 02_signal_record.py"
        )

    if not INPUT_METADATA.exists():
        raise FileNotFoundError(
            f"Adım 02 metadata dosyası bulunamadı: {INPUT_METADATA}"
        )

    with np.load(INPUT_PACKAGE, allow_pickle=False) as package:
        samples = package["samples"].astype(np.complex128)
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        sample_rate_hz = float(package["sample_rate_hz"])
        symbol_rate_hz = float(package["symbol_rate_hz"])
        samples_per_symbol = int(package["samples_per_symbol"])

    with INPUT_METADATA.open("r", encoding="utf-8") as file:
        source_metadata = json.load(file)

    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("Temiz sinyal tek boyutlu ve boş olmayan olmalıdır.")

    if not np.all(np.isfinite(samples.real)):
        raise ValueError("Temiz sinyalin gerçek kısmında NaN veya Inf bulundu.")

    if not np.all(np.isfinite(samples.imag)):
        raise ValueError("Temiz sinyalin sanal kısmında NaN veya Inf bulundu.")

    return CleanSignalPackage(
        samples=samples,
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
        source_metadata=source_metadata,
    )


def apply_carrier_impairment(
    samples: ComplexArray,
    sample_rate_hz: float,
    cfo_hz: float,
    phase_offset_degrees: float,
) -> tuple[ComplexArray, FloatArray]:
    """CFO ve sabit faz kayması uygular."""

    sample_indices = np.arange(len(samples), dtype=np.float64)
    phase_offset_radians = np.deg2rad(phase_offset_degrees)

    phase_trajectory = (
        2.0 * np.pi * cfo_hz * sample_indices / sample_rate_hz
        + phase_offset_radians
    )

    output = samples * np.exp(1j * phase_trajectory)

    return (
        output.astype(np.complex128),
        phase_trajectory.astype(np.float64),
    )


def build_multipath_channel(config: ImpairmentConfig) -> ComplexArray:
    """Enerjisi bire normalize edilmiş seyrek kompleks FIR kanal üretir."""

    channel = np.zeros(
        max(config.path_delays_samples) + 1,
        dtype=np.complex128,
    )

    for delay, magnitude, phase_degrees in zip(
        config.path_delays_samples,
        config.path_magnitudes,
        config.path_phases_degrees,
        strict=True,
    ):
        channel[delay] = magnitude * np.exp(
            1j * np.deg2rad(phase_degrees)
        )

    energy = float(np.sum(np.abs(channel) ** 2))

    if energy <= 0.0 or not np.isfinite(energy):
        raise RuntimeError("Geçerli bir multipath kanal oluşturulamadı.")

    channel /= np.sqrt(energy)
    return channel.astype(np.complex128)


def design_fractional_delay_filter(
    delay_samples: float,
    number_of_taps: int,
) -> FloatArray:
    """Pencerelenmiş sinc tabanlı fractional-delay FIR filtresi üretir."""

    half = (number_of_taps - 1) // 2
    indices = np.arange(-half, half + 1, dtype=np.float64)

    taps = np.sinc(indices - delay_samples)
    taps *= np.hamming(number_of_taps)

    dc_gain = float(np.sum(taps))

    if np.isclose(dc_gain, 0.0, atol=1e-15):
        raise RuntimeError("Fractional-delay filtresinin DC kazancı sıfır.")

    taps /= dc_gain
    return taps.astype(np.float64)


def apply_iq_imbalance(
    samples: ComplexArray,
    gain_imbalance_db: float,
    phase_imbalance_degrees: float,
) -> tuple[ComplexArray, float]:
    """IQ gain/phase imbalance uygular ve teorik IRR tahmini döndürür."""

    gain_ratio = 10.0 ** (gain_imbalance_db / 20.0)
    i_gain = np.sqrt(gain_ratio)
    q_gain = 1.0 / np.sqrt(gain_ratio)

    phase_error = np.deg2rad(phase_imbalance_degrees)
    half_phase = phase_error / 2.0

    i_axis = np.exp(1j * half_phase)
    q_axis = 1j * np.exp(-1j * half_phase)

    output = (
        samples.real * i_gain * i_axis
        + samples.imag * q_gain * q_axis
    ).astype(np.complex128)

    alpha = 0.5 * (i_gain * i_axis - 1j * q_gain * q_axis)
    beta = 0.5 * (i_gain * i_axis + 1j * q_gain * q_axis)

    if np.abs(beta) <= 0.0:
        irr_db = float("inf")
    else:
        irr_db = float(
            20.0 * np.log10(np.abs(alpha) / np.abs(beta))
        )

    return output, irr_db


def add_dc_offset(
    samples: ComplexArray,
    i_rms_ratio: float,
    q_rms_ratio: float,
) -> tuple[ComplexArray, complex]:
    """Giriş RMS genliğine göre kompleks DC offset ekler."""

    rms_magnitude = float(
        np.sqrt(np.mean(np.abs(samples) ** 2))
    )

    if rms_magnitude <= 0.0:
        raise ValueError("Giriş RMS genliği pozitif olmalıdır.")

    offset = rms_magnitude * complex(i_rms_ratio, q_rms_ratio)
    output = (samples + offset).astype(np.complex128)

    return output, offset


def hard_clip_complex(
    samples: ComplexArray,
    level_rms_multiplier: float,
) -> tuple[ComplexArray, float, float]:
    """Fazı koruyan kompleks hard clipping uygular."""

    rms_magnitude = float(
        np.sqrt(np.mean(np.abs(samples) ** 2))
    )
    clipping_level = rms_magnitude * level_rms_multiplier

    magnitudes = np.abs(samples)
    clipped_mask = magnitudes > clipping_level

    scale = np.ones_like(magnitudes, dtype=np.float64)
    valid_mask = clipped_mask & (magnitudes > 0.0)
    scale[valid_mask] = clipping_level / magnitudes[valid_mask]

    output = (samples * scale).astype(np.complex128)
    clipped_fraction = float(np.mean(clipped_mask))

    return output, clipping_level, clipped_fraction


def add_complex_awgn(
    samples: ComplexArray,
    target_snr_db: float,
    rng: np.random.Generator,
) -> tuple[ComplexArray, ComplexArray, float]:
    """Hedef SNR seviyesinde kompleks AWGN ekler."""

    signal_power = float(np.mean(np.abs(samples) ** 2))

    if signal_power <= 0.0:
        raise ValueError("Sinyal gücü pozitif olmalıdır.")

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    target_noise_power = signal_power / snr_linear
    component_std = np.sqrt(target_noise_power / 2.0)

    noise = component_std * (
        rng.standard_normal(len(samples))
        + 1j * rng.standard_normal(len(samples))
    )
    noise = noise.astype(np.complex128)

    output = (samples + noise).astype(np.complex128)

    achieved_noise_power = float(np.mean(np.abs(noise) ** 2))
    achieved_snr_db = float(
        10.0 * np.log10(signal_power / achieved_noise_power)
    )

    return output, noise, achieved_snr_db


def run_pipeline(
    clean_samples: ComplexArray,
    sample_rate_hz: float,
    config: ImpairmentConfig,
) -> PipelineResult:
    """Bütün bozunum zincirini tanımlanan sırayla çalıştırır."""

    config.validate(sample_rate_hz)
    rng = np.random.default_rng(config.random_seed)

    carrier_impaired, phase_trajectory = apply_carrier_impairment(
        samples=clean_samples,
        sample_rate_hz=sample_rate_hz,
        cfo_hz=config.cfo_hz,
        phase_offset_degrees=config.phase_offset_degrees,
    )

    channel_taps = build_multipath_channel(config)

    multipath_samples = np.convolve(
        carrier_impaired,
        channel_taps,
        mode="full",
    ).astype(np.complex128)

    fractional_delay_taps = design_fractional_delay_filter(
        delay_samples=config.fractional_delay_samples,
        number_of_taps=config.fractional_delay_filter_taps,
    )

    timing_impaired = np.convolve(
        multipath_samples,
        fractional_delay_taps,
        mode="full",
    ).astype(np.complex128)

    iq_impaired, iq_image_rejection_db = apply_iq_imbalance(
        samples=timing_impaired,
        gain_imbalance_db=config.iq_gain_imbalance_db,
        phase_imbalance_degrees=config.iq_phase_imbalance_degrees,
    )

    dc_offset_samples, dc_offset = add_dc_offset(
        samples=iq_impaired,
        i_rms_ratio=config.dc_offset_i_rms_ratio,
        q_rms_ratio=config.dc_offset_q_rms_ratio,
    )

    clipped_samples, clipping_level, clipped_fraction = (
        hard_clip_complex(
            samples=dc_offset_samples,
            level_rms_multiplier=(
                config.clipping_level_rms_multiplier
            ),
        )
    )

    received_samples, noise_samples, achieved_snr_db = (
        add_complex_awgn(
            samples=clipped_samples,
            target_snr_db=config.target_snr_db,
            rng=rng,
        )
    )

    return PipelineResult(
        clean_samples=clean_samples,
        carrier_impaired_samples=carrier_impaired,
        multipath_samples=multipath_samples,
        timing_impaired_samples=timing_impaired,
        iq_impaired_samples=iq_impaired,
        dc_offset_samples=dc_offset_samples,
        clipped_samples=clipped_samples,
        received_samples=received_samples,
        noise_samples=noise_samples,
        channel_taps=channel_taps,
        fractional_delay_taps=fractional_delay_taps,
        phase_trajectory_radians=phase_trajectory,
        dc_offset=dc_offset,
        clipping_level=clipping_level,
        clipped_fraction=clipped_fraction,
        achieved_snr_db=achieved_snr_db,
        iq_image_rejection_db=iq_image_rejection_db,
    )


def validate_pipeline_result(
    result: PipelineResult,
    config: ImpairmentConfig,
) -> dict[str, Any]:
    """Pipeline sonucunun fiziksel ve sayısal tutarlılığını doğrular."""

    arrays = {
        "clean": result.clean_samples,
        "carrier": result.carrier_impaired_samples,
        "multipath": result.multipath_samples,
        "timing": result.timing_impaired_samples,
        "iq": result.iq_impaired_samples,
        "dc": result.dc_offset_samples,
        "clipped": result.clipped_samples,
        "received": result.received_samples,
        "noise": result.noise_samples,
    }

    for name, array in arrays.items():
        if array.ndim != 1 or len(array) == 0:
            raise RuntimeError(f"{name} dizisinin boyutu geçersiz.")

        if not np.all(np.isfinite(array.real)):
            raise RuntimeError(f"{name} gerçek kısmında NaN veya Inf bulundu.")

        if not np.all(np.isfinite(array.imag)):
            raise RuntimeError(f"{name} sanal kısmında NaN veya Inf bulundu.")

    expected_multipath_length = (
        len(result.clean_samples) + len(result.channel_taps) - 1
    )
    expected_timing_length = (
        expected_multipath_length
        + len(result.fractional_delay_taps)
        - 1
    )

    if len(result.multipath_samples) != expected_multipath_length:
        raise RuntimeError("Multipath çıkış uzunluğu hatalı.")

    later_lengths = (
        len(result.timing_impaired_samples),
        len(result.iq_impaired_samples),
        len(result.dc_offset_samples),
        len(result.clipped_samples),
        len(result.received_samples),
        len(result.noise_samples),
    )

    if any(length != expected_timing_length for length in later_lengths):
        raise RuntimeError("Pipeline'ın sonraki aşamalarında uzunluk hatası var.")

    channel_energy = float(
        np.sum(np.abs(result.channel_taps) ** 2)
    )
    delay_dc_gain = float(
        np.sum(result.fractional_delay_taps)
    )

    if not np.isclose(channel_energy, 1.0, atol=1e-12):
        raise RuntimeError("Multipath kanal enerjisi bire eşit değil.")

    if not np.isclose(delay_dc_gain, 1.0, atol=1e-12):
        raise RuntimeError("Fractional-delay filtresi DC kazancı bire eşit değil.")

    maximum_clipped_magnitude = float(
        np.max(np.abs(result.clipped_samples))
    )

    if maximum_clipped_magnitude > result.clipping_level + 1e-12:
        raise RuntimeError("Clipping seviyesi aşıldı.")

    snr_error_db = abs(
        result.achieved_snr_db - config.target_snr_db
    )

    if snr_error_db > 0.25:
        raise RuntimeError("Gerçekleşen SNR tolerans dışında.")

    return {
        "expected_multipath_length": expected_multipath_length,
        "expected_output_length": expected_timing_length,
        "channel_energy": channel_energy,
        "fractional_delay_dc_gain": delay_dc_gain,
        "maximum_clipped_magnitude": maximum_clipped_magnitude,
        "snr_error_db": snr_error_db,
    }


def verify_reproducibility(
    package: CleanSignalPackage,
    config: ImpairmentConfig,
    first_result: PipelineResult,
) -> str:
    """Aynı seed ve config ile ikinci çalışmanın aynı çıktıyı verdiğini doğrular."""

    second_result = run_pipeline(
        clean_samples=package.samples,
        sample_rate_hz=package.sample_rate_hz,
        config=config,
    )

    if not np.array_equal(
        first_result.received_samples,
        second_result.received_samples,
    ):
        raise RuntimeError(
            "Reproducibility testi başarısız: received_samples farklı."
        )

    if not np.array_equal(
        first_result.noise_samples,
        second_result.noise_samples,
    ):
        raise RuntimeError(
            "Reproducibility testi başarısız: noise_samples farklı."
        )

    return array_sha256(first_result.received_samples)


def calculate_power_spectrum_db(
    samples: ComplexArray,
    sample_rate_hz: float,
    fft_size: int = 16384,
) -> tuple[FloatArray, FloatArray]:
    """Normalize güç spektrumu hesaplar."""

    segment_length = min(len(samples), fft_size)
    segment = samples[:segment_length]
    windowed = segment * np.hanning(segment_length)

    spectrum = np.fft.fftshift(
        np.fft.fft(windowed, n=fft_size)
    )
    power = np.abs(spectrum) ** 2
    maximum = float(np.max(power))

    if maximum > 0.0:
        power /= maximum

    power_db = 10.0 * np.log10(np.maximum(power, 1e-15))
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz)
    )

    return (
        frequencies.astype(np.float64),
        power_db.astype(np.float64),
    )


def create_overview_plot(
    result: PipelineResult,
    sample_rate_hz: float,
    output_path: Path,
) -> None:
    """Pipeline'ın başlıca aşamalarını tek görselde gösterir."""

    figure, axes = plt.subplots(4, 1, figsize=(12, 17))

    stage_names = [
        "Clean",
        "Carrier",
        "Multipath",
        "Timing",
        "IQ",
        "DC",
        "Clipping",
        "AWGN",
    ]

    stage_powers = [
        float(np.mean(np.abs(result.clean_samples) ** 2)),
        float(np.mean(np.abs(result.carrier_impaired_samples) ** 2)),
        float(np.mean(np.abs(result.multipath_samples) ** 2)),
        float(np.mean(np.abs(result.timing_impaired_samples) ** 2)),
        float(np.mean(np.abs(result.iq_impaired_samples) ** 2)),
        float(np.mean(np.abs(result.dc_offset_samples) ** 2)),
        float(np.mean(np.abs(result.clipped_samples) ** 2)),
        float(np.mean(np.abs(result.received_samples) ** 2)),
    ]

    axes[0].bar(stage_names, stage_powers)
    axes[0].set_title("Pipeline Aşamalarına Göre Ortalama Güç")
    axes[0].set_ylabel("Ortalama güç")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y")

    plot_count = min(
        2500,
        len(result.timing_impaired_samples),
        len(result.received_samples),
    )

    axes[1].scatter(
        result.timing_impaired_samples[:plot_count].real,
        result.timing_impaired_samples[:plot_count].imag,
        s=5,
        alpha=0.25,
        label="IQ/DC/Clipping öncesi",
    )
    axes[1].scatter(
        result.received_samples[:plot_count].real,
        result.received_samples[:plot_count].imag,
        s=5,
        alpha=0.25,
        label="Nihai alınan sinyal",
    )
    axes[1].set_title("Kompleks Düzlem Karşılaştırması")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True)
    axes[1].legend()

    clean_frequency, clean_spectrum = calculate_power_spectrum_db(
        result.clean_samples,
        sample_rate_hz,
    )
    received_frequency, received_spectrum = calculate_power_spectrum_db(
        result.received_samples,
        sample_rate_hz,
    )

    axes[2].plot(
        clean_frequency / 1_000.0,
        clean_spectrum,
        label="Temiz sinyal",
    )
    axes[2].plot(
        received_frequency / 1_000.0,
        received_spectrum,
        label="Nihai bozulmuş sinyal",
        alpha=0.8,
    )
    axes[2].set_title("Normalize Güç Spektrumu")
    axes[2].set_xlabel("Frekans (kHz)")
    axes[2].set_ylabel("Normalize güç (dB)")
    axes[2].set_ylim(-100.0, 5.0)
    axes[2].grid(True)
    axes[2].legend()

    magnitude_before = np.abs(result.dc_offset_samples)
    magnitude_after = np.abs(result.clipped_samples)

    axes[3].hist(
        magnitude_before,
        bins=100,
        density=True,
        alpha=0.55,
        label="Clipping öncesi",
    )
    axes[3].hist(
        magnitude_after,
        bins=100,
        density=True,
        alpha=0.55,
        label="Clipping sonrası",
    )
    axes[3].axvline(
        result.clipping_level,
        linestyle="--",
        label="Clipping seviyesi",
    )
    axes[3].set_title("Genlik Dağılımı")
    axes[3].set_xlabel("Kompleks genlik")
    axes[3].set_ylabel("Yoğunluk")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_outputs(
    package: CleanSignalPackage,
    config: ImpairmentConfig,
    result: PipelineResult,
    validations: dict[str, Any],
    output_sha256: str,
) -> None:
    """Pipeline çıktılarının tamamını kaydeder."""

    STEP06_DIRECTORY.mkdir(parents=True, exist_ok=True)

    config_document = {
        "schema_name": "GENESIS-DSP ImpairmentConfig",
        "schema_version": "1.0.0",
        "config": asdict(config),
        "processing_order": [
            "carrier_frequency_offset",
            "constant_phase_offset",
            "multipath_fir_channel",
            "fractional_timing_delay",
            "iq_gain_phase_imbalance",
            "complex_dc_offset",
            "hard_magnitude_clipping",
            "complex_awgn",
        ],
    }

    with (STEP06_DIRECTORY / "pipeline_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_document,
            file,
            indent=4,
            ensure_ascii=False,
        )

    np.savez_compressed(
        STEP06_DIRECTORY / "dataset_case_0001.npz",
        clean_samples=result.clean_samples,
        carrier_impaired_samples=result.carrier_impaired_samples,
        multipath_samples=result.multipath_samples,
        timing_impaired_samples=result.timing_impaired_samples,
        iq_impaired_samples=result.iq_impaired_samples,
        dc_offset_samples=result.dc_offset_samples,
        clipped_samples=result.clipped_samples,
        received_samples=result.received_samples,
        noise_samples=result.noise_samples,
        channel_taps=result.channel_taps,
        fractional_delay_taps=result.fractional_delay_taps,
        phase_trajectory_radians=result.phase_trajectory_radians,
        bits=package.bits,
        symbols=package.symbols,
        sample_rate_hz=np.float64(package.sample_rate_hz),
        symbol_rate_hz=np.float64(package.symbol_rate_hz),
        samples_per_symbol=np.int64(package.samples_per_symbol),
        dc_offset=np.complex128(result.dc_offset),
        clipping_level=np.float64(result.clipping_level),
        clipped_fraction=np.float64(result.clipped_fraction),
        achieved_snr_db=np.float64(result.achieved_snr_db),
        iq_image_rejection_db=np.float64(
            result.iq_image_rejection_db
        ),
    )

    metadata = {
        "project": "GENESIS-DSP",
        "step": 6,
        "description": "Reusable ordered composite impairment pipeline",
        "input_step": 2,
        "input_samples_sha256": array_sha256(package.samples),
        "output_samples_sha256": output_sha256,
        "random_seed": config.random_seed,
        "signal": {
            "input_number_of_samples": int(len(package.samples)),
            "output_number_of_samples": int(
                len(result.received_samples)
            ),
            "sample_rate_hz": package.sample_rate_hz,
            "symbol_rate_hz": package.symbol_rate_hz,
            "samples_per_symbol": package.samples_per_symbol,
        },
        "results": {
            "achieved_snr_db": result.achieved_snr_db,
            "snr_error_db": validations["snr_error_db"],
            "iq_image_rejection_db": (
                result.iq_image_rejection_db
            ),
            "dc_offset_real": float(result.dc_offset.real),
            "dc_offset_imag": float(result.dc_offset.imag),
            "clipping_level": result.clipping_level,
            "clipped_fraction": result.clipped_fraction,
            "clipped_percentage": (
                100.0 * result.clipped_fraction
            ),
            "channel_energy": validations["channel_energy"],
            "fractional_delay_dc_gain": (
                validations["fractional_delay_dc_gain"]
            ),
        },
        "validations": {
            **validations,
            "finite_all_stages": True,
            "reproducibility_test": "PASSED",
            "same_seed_same_output": True,
        },
    }

    with (STEP06_DIRECTORY / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    create_overview_plot(
        result=result,
        sample_rate_hz=package.sample_rate_hz,
        output_path=STEP06_DIRECTORY / "pipeline_overview.png",
    )


def main() -> None:
    package = load_clean_signal()
    config = ImpairmentConfig()

    result = run_pipeline(
        clean_samples=package.samples,
        sample_rate_hz=package.sample_rate_hz,
        config=config,
    )

    validations = validate_pipeline_result(
        result=result,
        config=config,
    )

    output_sha256 = verify_reproducibility(
        package=package,
        config=config,
        first_result=result,
    )

    save_outputs(
        package=package,
        config=config,
        result=result,
        validations=validations,
        output_sha256=output_sha256,
    )

    print()
    print("=" * 76)
    print("GENESIS-DSP — ADIM 06 BAŞARIYLA TAMAMLANDI")
    print("=" * 76)
    print(f"Giriş örnek sayısı              : {len(package.samples)}")
    print(f"Çıkış örnek sayısı              : {len(result.received_samples)}")
    print(f"Uygulanan bozunum sayısı        : 8")
    print(f"CFO                              : {config.cfo_hz:,.6f} Hz")
    print(
        f"Fractional timing               : "
        f"{config.fractional_delay_samples:.6f} örnek"
    )
    print(
        f"IQ image rejection              : "
        f"{result.iq_image_rejection_db:.6f} dB"
    )
    print(
        f"Clipping oranı                  : "
        f"{100.0 * result.clipped_fraction:.6f} %"
    )
    print(f"Hedef SNR                       : {config.target_snr_db:.6f} dB")
    print(f"Gerçekleşen SNR                 : {result.achieved_snr_db:.6f} dB")
    print(f"SNR hatası                      : {validations['snr_error_db']:.6f} dB")
    print(f"Çıkış SHA-256                   : {output_sha256}")
    print("Reproducibility testi           : BAŞARILI")
    print(f"Sonuç klasörü                   : {STEP06_DIRECTORY}")
    print("=" * 76)


if __name__ == "__main__":
    main()
