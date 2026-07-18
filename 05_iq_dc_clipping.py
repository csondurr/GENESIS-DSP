

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP04_DIRECTORY = BASE_DIRECTORY / "outputs" / "step04"
STEP05_DIRECTORY = BASE_DIRECTORY / "outputs" / "step05"

INPUT_PACKAGE = STEP04_DIRECTORY / "channel_timing_record.npz"
INPUT_METADATA = STEP04_DIRECTORY / "metadata.json"

RANDOM_SEED = 20260716 + 5

# IQ dengesizliği
IQ_GAIN_IMBALANCE_DB = 1.20
IQ_PHASE_IMBALANCE_DEGREES = 5.0

# Kompleks DC offset, giriş RMS genliğinin oranı olarak tanımlanır.
DC_OFFSET_I_RMS_RATIO = 0.12
DC_OFFSET_Q_RMS_RATIO = -0.08

# Hard clipping seviyesi, clipping öncesi RMS genliğinin katıdır.
CLIPPING_LEVEL_RMS_MULTIPLIER = 1.25

DEFAULT_TARGET_SNR_DB = 12.0

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Step04Package:
    clean_samples: ComplexArray
    carrier_impaired_samples: ComplexArray
    channel_output: ComplexArray
    noiseless_received_samples: ComplexArray
    bits: NDArray[np.int8]
    symbols: ComplexArray
    channel_taps: ComplexArray
    fractional_delay_taps: FloatArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    cfo_hz: float
    phase_offset_degrees: float
    fractional_delay_samples: float
    target_snr_db: float
    metadata: dict[str, Any]


def load_step04_package() -> Step04Package:
    """Adım 04 veri paketini ve metadata dosyasını yükler."""

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 04 veri paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python 04_multipath_timing.py"
        )

    if not INPUT_METADATA.exists():
        raise FileNotFoundError(
            f"Adım 04 metadata dosyası bulunamadı: {INPUT_METADATA}"
        )

    with np.load(INPUT_PACKAGE, allow_pickle=False) as package:
        clean_samples = package["clean_samples"].astype(np.complex128)
        carrier_impaired_samples = package[
            "carrier_impaired_samples"
        ].astype(np.complex128)
        channel_output = package["channel_output"].astype(np.complex128)
        noiseless_received_samples = package[
            "noiseless_received_samples"
        ].astype(np.complex128)
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        channel_taps = package["channel_taps"].astype(np.complex128)
        fractional_delay_taps = package[
            "fractional_delay_taps"
        ].astype(np.float64)
        sample_rate_hz = float(package["sample_rate_hz"])
        symbol_rate_hz = float(package["symbol_rate_hz"])
        samples_per_symbol = int(package["samples_per_symbol"])
        cfo_hz = float(package["cfo_hz"])
        phase_offset_degrees = float(package["phase_offset_degrees"])
        fractional_delay_samples = float(
            package["fractional_delay_samples"]
        )

        if "target_snr_db" in package.files:
            target_snr_db = float(package["target_snr_db"])
        else:
            target_snr_db = DEFAULT_TARGET_SNR_DB

    with INPUT_METADATA.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    complex_arrays = {
        "clean_samples": clean_samples,
        "carrier_impaired_samples": carrier_impaired_samples,
        "channel_output": channel_output,
        "noiseless_received_samples": noiseless_received_samples,
        "symbols": symbols,
        "channel_taps": channel_taps,
    }

    for name, array in complex_arrays.items():
        if array.ndim != 1 or len(array) == 0:
            raise ValueError(f"{name} tek boyutlu ve boş olmayan olmalıdır.")

        if not np.all(np.isfinite(array.real)):
            raise ValueError(f"{name} gerçek kısmında NaN veya Inf bulundu.")

        if not np.all(np.isfinite(array.imag)):
            raise ValueError(f"{name} sanal kısmında NaN veya Inf bulundu.")

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz pozitif olmalıdır.")

    if symbol_rate_hz <= 0.0:
        raise ValueError("symbol_rate_hz pozitif olmalıdır.")

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol pozitif olmalıdır.")

    return Step04Package(
        clean_samples=clean_samples,
        carrier_impaired_samples=carrier_impaired_samples,
        channel_output=channel_output,
        noiseless_received_samples=noiseless_received_samples,
        bits=bits,
        symbols=symbols,
        channel_taps=channel_taps,
        fractional_delay_taps=fractional_delay_taps,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
        cfo_hz=cfo_hz,
        phase_offset_degrees=phase_offset_degrees,
        fractional_delay_samples=fractional_delay_samples,
        target_snr_db=target_snr_db,
        metadata=metadata,
    )


def apply_iq_imbalance(
    samples: ComplexArray,
    gain_imbalance_db: float,
    phase_imbalance_degrees: float,
) -> tuple[ComplexArray, dict[str, float]]:
    """
    I ve Q kollarına gain ve quadrature phase imbalance uygular.

    Gain imbalance:
        I kolu sqrt(gain_ratio) ile,
        Q kolu 1/sqrt(gain_ratio) ile ölçeklenir.

    Phase imbalance:
        I ve Q eksenleri ideal 90 derece yerine simetrik olarak kaydırılır.
    """

    if not np.isfinite(gain_imbalance_db):
        raise ValueError("gain_imbalance_db sonlu olmalıdır.")

    if not np.isfinite(phase_imbalance_degrees):
        raise ValueError("phase_imbalance_degrees sonlu olmalıdır.")

    gain_ratio = 10.0 ** (gain_imbalance_db / 20.0)
    i_gain = np.sqrt(gain_ratio)
    q_gain = 1.0 / np.sqrt(gain_ratio)

    phase_error_rad = np.deg2rad(phase_imbalance_degrees)
    half_phase = phase_error_rad / 2.0

    i_component = samples.real * i_gain
    q_component = samples.imag * q_gain

    i_axis = np.exp(1j * half_phase)
    q_axis = 1j * np.exp(-1j * half_phase)

    output = (
        i_component * i_axis
        + q_component * q_axis
    ).astype(np.complex128)

    # y = alpha*x + beta*conj(x) eşdeğer katsayıları
    alpha = 0.5 * (i_gain * i_axis - 1j * q_gain * q_axis)
    beta = 0.5 * (i_gain * i_axis + 1j * q_gain * q_axis)

    beta_magnitude = float(np.abs(beta))
    alpha_magnitude = float(np.abs(alpha))

    if beta_magnitude <= 0.0:
        image_rejection_ratio_db = float("inf")
    else:
        image_rejection_ratio_db = float(
            20.0 * np.log10(alpha_magnitude / beta_magnitude)
        )

    diagnostics = {
        "gain_ratio_linear": float(gain_ratio),
        "i_branch_gain": float(i_gain),
        "q_branch_gain": float(q_gain),
        "phase_error_radians": float(phase_error_rad),
        "alpha_real": float(alpha.real),
        "alpha_imag": float(alpha.imag),
        "beta_real": float(beta.real),
        "beta_imag": float(beta.imag),
        "image_rejection_ratio_db": image_rejection_ratio_db,
    }

    return output, diagnostics


def add_dc_offset(
    samples: ComplexArray,
    i_rms_ratio: float,
    q_rms_ratio: float,
) -> tuple[ComplexArray, complex]:
    """
    Giriş RMS genliğine göre kompleks DC offset ekler.
    """

    if not np.isfinite(i_rms_ratio) or not np.isfinite(q_rms_ratio):
        raise ValueError("DC offset oranları sonlu olmalıdır.")

    rms_magnitude = float(
        np.sqrt(np.mean(np.abs(samples) ** 2))
    )

    if rms_magnitude <= 0.0:
        raise ValueError("Giriş RMS genliği pozitif olmalıdır.")

    dc_offset = rms_magnitude * complex(i_rms_ratio, q_rms_ratio)
    output = (samples + dc_offset).astype(np.complex128)

    return output, dc_offset


def hard_clip_complex(
    samples: ComplexArray,
    level_rms_multiplier: float,
) -> tuple[ComplexArray, float, float]:
    """
    Kompleks örneklerin fazını koruyarak genliklerini sınırlar.

    |x[n]| > A ise:
        y[n] = A * exp(j*angle(x[n]))
    """

    if not np.isfinite(level_rms_multiplier):
        raise ValueError("level_rms_multiplier sonlu olmalıdır.")

    if level_rms_multiplier <= 0.0:
        raise ValueError("level_rms_multiplier pozitif olmalıdır.")

    rms_magnitude = float(
        np.sqrt(np.mean(np.abs(samples) ** 2))
    )
    clipping_level = level_rms_multiplier * rms_magnitude

    magnitudes = np.abs(samples)
    clipped_mask = magnitudes > clipping_level

    scale = np.ones_like(magnitudes, dtype=np.float64)
    nonzero_clipped = clipped_mask & (magnitudes > 0.0)
    scale[nonzero_clipped] = (
        clipping_level / magnitudes[nonzero_clipped]
    )

    clipped = (samples * scale).astype(np.complex128)
    clipped_fraction = float(np.mean(clipped_mask))

    return clipped, clipping_level, clipped_fraction


def add_complex_awgn(
    samples: ComplexArray,
    target_snr_db: float,
    rng: np.random.Generator,
) -> tuple[ComplexArray, ComplexArray, float]:
    """Hedef SNR seviyesinde kompleks AWGN ekler."""

    signal_power = float(np.mean(np.abs(samples) ** 2))

    if signal_power <= 0.0 or not np.isfinite(signal_power):
        raise ValueError("Sinyal gücü pozitif ve sonlu olmalıdır.")

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    target_noise_power = signal_power / snr_linear
    component_std = np.sqrt(target_noise_power / 2.0)

    noise = component_std * (
        rng.standard_normal(len(samples))
        + 1j * rng.standard_normal(len(samples))
    )
    noise = noise.astype(np.complex128)

    received = (samples + noise).astype(np.complex128)

    achieved_noise_power = float(np.mean(np.abs(noise) ** 2))
    achieved_snr_db = float(
        10.0 * np.log10(signal_power / achieved_noise_power)
    )

    return received, noise, achieved_snr_db


def calculate_nmse(
    reference: ComplexArray,
    estimate: ComplexArray,
) -> float:
    """Normalize edilmiş ortalama karesel hatayı hesaplar."""

    if reference.shape != estimate.shape:
        raise ValueError("reference ve estimate aynı boyutta olmalıdır.")

    denominator = float(np.sum(np.abs(reference) ** 2))

    if denominator <= 0.0:
        raise ValueError("Referans sinyal enerjisi pozitif olmalıdır.")

    numerator = float(np.sum(np.abs(reference - estimate) ** 2))
    return numerator / denominator


def calculate_power_spectrum_db(
    samples: ComplexArray,
    sample_rate_hz: float,
    fft_size: int = 16384,
) -> tuple[FloatArray, FloatArray]:
    """Normalize güç spektrumunu hesaplar."""

    segment_length = min(len(samples), fft_size)
    segment = samples[:segment_length]
    window = np.hanning(segment_length)

    spectrum = np.fft.fftshift(
        np.fft.fft(segment * window, n=fft_size)
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


def create_plots(
    input_samples: ComplexArray,
    iq_samples: ComplexArray,
    dc_samples: ComplexArray,
    clipped_samples: ComplexArray,
    received_samples: ComplexArray,
    sample_rate_hz: float,
    output_path: Path,
) -> None:
    """Bozunumların etkisini gösteren dört grafik üretir."""

    figure, axes = plt.subplots(4, 1, figsize=(12, 17))

    plot_count = min(2500, len(input_samples))

    axes[0].scatter(
        input_samples[:plot_count].real,
        input_samples[:plot_count].imag,
        s=5,
        alpha=0.25,
        label="Adım 04 girişi",
    )
    axes[0].scatter(
        iq_samples[:plot_count].real,
        iq_samples[:plot_count].imag,
        s=5,
        alpha=0.25,
        label="IQ imbalance sonrası",
    )
    axes[0].set_title("IQ Dengesizliği — Kompleks Düzlem")
    axes[0].set_xlabel("I")
    axes[0].set_ylabel("Q")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].scatter(
        dc_samples[:plot_count].real,
        dc_samples[:plot_count].imag,
        s=5,
        alpha=0.25,
        label="DC offset sonrası",
    )
    axes[1].scatter(
        received_samples[:plot_count].real,
        received_samples[:plot_count].imag,
        s=5,
        alpha=0.25,
        label="Nihai alınan sinyal",
    )
    axes[1].set_title("DC Offset ve Nihai Bozulmuş Sinyal")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True)
    axes[1].legend()

    before_magnitude = np.abs(dc_samples)
    after_magnitude = np.abs(clipped_samples)

    axes[2].hist(
        before_magnitude,
        bins=100,
        alpha=0.55,
        label="Clipping öncesi",
        density=True,
    )
    axes[2].hist(
        after_magnitude,
        bins=100,
        alpha=0.55,
        label="Clipping sonrası",
        density=True,
    )
    axes[2].set_title("Genlik Dağılımı ve Clipping")
    axes[2].set_xlabel("Kompleks genlik")
    axes[2].set_ylabel("Yoğunluk")
    axes[2].grid(True)
    axes[2].legend()

    input_frequency, input_spectrum = calculate_power_spectrum_db(
        input_samples,
        sample_rate_hz,
    )
    received_frequency, received_spectrum = calculate_power_spectrum_db(
        received_samples,
        sample_rate_hz,
    )

    axes[3].plot(
        input_frequency / 1_000.0,
        input_spectrum,
        label="Adım 04 girişi",
    )
    axes[3].plot(
        received_frequency / 1_000.0,
        received_spectrum,
        label="IQ + DC + Clipping + AWGN",
        alpha=0.8,
    )
    axes[3].set_title("Normalize Güç Spektrumu")
    axes[3].set_xlabel("Frekans (kHz)")
    axes[3].set_ylabel("Normalize Güç (dB)")
    axes[3].set_ylim(-100.0, 5.0)
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    STEP05_DIRECTORY.mkdir(parents=True, exist_ok=True)

    package = load_step04_package()
    rng = np.random.default_rng(RANDOM_SEED)

    input_samples = package.noiseless_received_samples

    iq_samples, iq_diagnostics = apply_iq_imbalance(
        samples=input_samples,
        gain_imbalance_db=IQ_GAIN_IMBALANCE_DB,
        phase_imbalance_degrees=IQ_PHASE_IMBALANCE_DEGREES,
    )

    dc_samples, dc_offset = add_dc_offset(
        samples=iq_samples,
        i_rms_ratio=DC_OFFSET_I_RMS_RATIO,
        q_rms_ratio=DC_OFFSET_Q_RMS_RATIO,
    )

    clipped_samples, clipping_level, clipped_fraction = (
        hard_clip_complex(
            samples=dc_samples,
            level_rms_multiplier=CLIPPING_LEVEL_RMS_MULTIPLIER,
        )
    )

    received_samples, noise_samples, achieved_snr_db = (
        add_complex_awgn(
            samples=clipped_samples,
            target_snr_db=package.target_snr_db,
            rng=rng,
        )
    )

    arrays_to_validate = {
        "iq_samples": iq_samples,
        "dc_samples": dc_samples,
        "clipped_samples": clipped_samples,
        "received_samples": received_samples,
        "noise_samples": noise_samples,
    }

    for name, array in arrays_to_validate.items():
        if len(array) != len(input_samples):
            raise RuntimeError(f"{name} sinyal uzunluğunu değiştirdi.")

        if not np.all(np.isfinite(array.real)):
            raise RuntimeError(f"{name} gerçek kısmında NaN veya Inf var.")

        if not np.all(np.isfinite(array.imag)):
            raise RuntimeError(f"{name} sanal kısmında NaN veya Inf var.")

    maximum_clipped_magnitude = float(
        np.max(np.abs(clipped_samples))
    )

    if maximum_clipped_magnitude > clipping_level + 1e-12:
        raise RuntimeError("Clipping seviyesi aşıldı.")

    if not 0.0 < clipped_fraction < 1.0:
        raise RuntimeError(
            "Clipping oranı geçersiz; parametreler kontrol edilmeli."
        )

    snr_error_db = abs(achieved_snr_db - package.target_snr_db)

    if snr_error_db > 0.25:
        raise RuntimeError("Gerçekleşen SNR tolerans dışında.")

    iq_nmse = calculate_nmse(
        reference=input_samples,
        estimate=iq_samples,
    )
    dc_nmse = calculate_nmse(
        reference=iq_samples,
        estimate=dc_samples,
    )
    clipping_nmse = calculate_nmse(
        reference=dc_samples,
        estimate=clipped_samples,
    )
    total_noiseless_nmse = calculate_nmse(
        reference=input_samples,
        estimate=clipped_samples,
    )

    np.save(STEP05_DIRECTORY / "input_samples.npy", input_samples)
    np.save(STEP05_DIRECTORY / "iq_impaired_samples.npy", iq_samples)
    np.save(STEP05_DIRECTORY / "dc_offset_samples.npy", dc_samples)
    np.save(STEP05_DIRECTORY / "clipped_samples.npy", clipped_samples)
    np.save(STEP05_DIRECTORY / "received_samples.npy", received_samples)
    np.save(STEP05_DIRECTORY / "noise_samples.npy", noise_samples)
    np.save(STEP05_DIRECTORY / "bits.npy", package.bits)
    np.save(STEP05_DIRECTORY / "symbols.npy", package.symbols)

    np.savez_compressed(
        STEP05_DIRECTORY / "frontend_impairments_record.npz",
        clean_samples=package.clean_samples,
        carrier_impaired_samples=package.carrier_impaired_samples,
        channel_output=package.channel_output,
        timing_impaired_samples=input_samples,
        iq_impaired_samples=iq_samples,
        dc_offset_samples=dc_samples,
        clipped_samples=clipped_samples,
        received_samples=received_samples,
        noise_samples=noise_samples,
        channel_taps=package.channel_taps,
        fractional_delay_taps=package.fractional_delay_taps,
        bits=package.bits,
        symbols=package.symbols,
        sample_rate_hz=np.float64(package.sample_rate_hz),
        symbol_rate_hz=np.float64(package.symbol_rate_hz),
        samples_per_symbol=np.int64(package.samples_per_symbol),
        cfo_hz=np.float64(package.cfo_hz),
        phase_offset_degrees=np.float64(
            package.phase_offset_degrees
        ),
        fractional_delay_samples=np.float64(
            package.fractional_delay_samples
        ),
        iq_gain_imbalance_db=np.float64(
            IQ_GAIN_IMBALANCE_DB
        ),
        iq_phase_imbalance_degrees=np.float64(
            IQ_PHASE_IMBALANCE_DEGREES
        ),
        dc_offset=np.complex128(dc_offset),
        clipping_level=np.float64(clipping_level),
        clipping_level_rms_multiplier=np.float64(
            CLIPPING_LEVEL_RMS_MULTIPLIER
        ),
        clipped_fraction=np.float64(clipped_fraction),
        target_snr_db=np.float64(package.target_snr_db),
        achieved_snr_db=np.float64(achieved_snr_db),
    )

    metadata = {
        "project": "GENESIS-DSP",
        "step": 5,
        "description": (
            "IQ imbalance, DC offset, hard clipping and AWGN engine"
        ),
        "random_seed": RANDOM_SEED,
        "input_step": 4,
        "signal": {
            "number_of_samples": int(len(input_samples)),
            "sample_rate_hz": package.sample_rate_hz,
            "symbol_rate_hz": package.symbol_rate_hz,
            "samples_per_symbol": package.samples_per_symbol,
        },
        "iq_imbalance": {
            "gain_imbalance_db": IQ_GAIN_IMBALANCE_DB,
            "phase_imbalance_degrees": (
                IQ_PHASE_IMBALANCE_DEGREES
            ),
            **iq_diagnostics,
            "nmse": iq_nmse,
        },
        "dc_offset": {
            "i_rms_ratio": DC_OFFSET_I_RMS_RATIO,
            "q_rms_ratio": DC_OFFSET_Q_RMS_RATIO,
            "offset_real": float(dc_offset.real),
            "offset_imag": float(dc_offset.imag),
            "nmse": dc_nmse,
        },
        "clipping": {
            "level_rms_multiplier": (
                CLIPPING_LEVEL_RMS_MULTIPLIER
            ),
            "absolute_level": clipping_level,
            "maximum_output_magnitude": (
                maximum_clipped_magnitude
            ),
            "clipped_fraction": clipped_fraction,
            "clipped_percentage": 100.0 * clipped_fraction,
            "nmse": clipping_nmse,
        },
        "awgn": {
            "target_snr_db": package.target_snr_db,
            "achieved_snr_db": achieved_snr_db,
            "snr_error_db": snr_error_db,
            "signal_power_before_noise": float(
                np.mean(np.abs(clipped_samples) ** 2)
            ),
            "noise_power": float(
                np.mean(np.abs(noise_samples) ** 2)
            ),
        },
        "measurements": {
            "input_power": float(
                np.mean(np.abs(input_samples) ** 2)
            ),
            "iq_output_power": float(
                np.mean(np.abs(iq_samples) ** 2)
            ),
            "dc_output_power": float(
                np.mean(np.abs(dc_samples) ** 2)
            ),
            "clipped_output_power": float(
                np.mean(np.abs(clipped_samples) ** 2)
            ),
            "received_power": float(
                np.mean(np.abs(received_samples) ** 2)
            ),
            "total_noiseless_nmse": total_noiseless_nmse,
        },
        "processing_order": [
            "clean QPSK with RRC pulse shaping",
            "carrier frequency offset",
            "constant phase offset",
            "multipath FIR channel",
            "fractional timing delay",
            "IQ gain and phase imbalance",
            "complex DC offset",
            "hard magnitude clipping",
            "complex AWGN",
        ],
        "validations": {
            "all_lengths_preserved": True,
            "finite_outputs": True,
            "clipping_limit_respected": True,
            "nonzero_clipping_fraction": True,
            "snr_tolerance_passed": True,
        },
    }

    with (STEP05_DIRECTORY / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    create_plots(
        input_samples=input_samples,
        iq_samples=iq_samples,
        dc_samples=dc_samples,
        clipped_samples=clipped_samples,
        received_samples=received_samples,
        sample_rate_hz=package.sample_rate_hz,
        output_path=STEP05_DIRECTORY / "frontend_impairments_overview.png",
    )

    print()
    print("=" * 74)
    print("GENESIS-DSP — ADIM 05 BAŞARIYLA TAMAMLANDI")
    print("=" * 74)
    print(f"Örnek sayısı                    : {len(input_samples)}")
    print(f"IQ gain imbalance              : {IQ_GAIN_IMBALANCE_DB:.6f} dB")
    print(
        f"IQ phase imbalance             : "
        f"{IQ_PHASE_IMBALANCE_DEGREES:.6f} derece"
    )
    print(
        f"Tahmini image rejection        : "
        f"{iq_diagnostics['image_rejection_ratio_db']:.6f} dB"
    )
    print(
        f"DC offset                      : "
        f"{dc_offset.real:+.8f} {dc_offset.imag:+.8f}j"
    )
    print(f"Clipping seviyesi              : {clipping_level:.12f}")
    print(
        f"Clipping oranı                 : "
        f"{100.0 * clipped_fraction:.6f} %"
    )
    print(f"Hedef SNR                      : {package.target_snr_db:.6f} dB")
    print(f"Gerçekleşen SNR                : {achieved_snr_db:.6f} dB")
    print(f"SNR hatası                     : {snr_error_db:.6f} dB")
    print(f"Toplam gürültüsüz NMSE         : {total_noiseless_nmse:.12e}")
    print(f"Sonuç klasörü                  : {STEP05_DIRECTORY}")
    print("=" * 74)


if __name__ == "__main__":
    main()
