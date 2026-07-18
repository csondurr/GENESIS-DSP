

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP02_DIRECTORY = BASE_DIRECTORY / "outputs" / "step02"
STEP03_DIRECTORY = BASE_DIRECTORY / "outputs" / "step03"

INPUT_PACKAGE = STEP02_DIRECTORY / "signal_record.npz"
INPUT_METADATA = STEP02_DIRECTORY / "signal_record_metadata.json"

RANDOM_SEED = 20260716
CFO_HZ = 7_500.0
PHASE_OFFSET_DEGREES = 32.0
TARGET_SNR_DB = 12.0

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SignalPackage:
    samples: ComplexArray
    bits: NDArray[np.int8]
    symbols: ComplexArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    metadata: dict[str, Any]


def load_step02_package() -> SignalPackage:
    """Adım 02 tarafından üretilen standart sinyal paketini yükler."""

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 02 veri paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python 02_signal_record.py"
        )

    if not INPUT_METADATA.exists():
        raise FileNotFoundError(
            f"Adım 02 metadata dosyası bulunamadı: {INPUT_METADATA}\n"
            "Önce şu komutu çalıştır:\n"
            "python 02_signal_record.py"
        )

    with np.load(INPUT_PACKAGE, allow_pickle=False) as package:
        samples = package["samples"].astype(np.complex128)
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        sample_rate_hz = float(package["sample_rate_hz"])
        symbol_rate_hz = float(package["symbol_rate_hz"])
        samples_per_symbol = int(package["samples_per_symbol"])

    with INPUT_METADATA.open("r", encoding="utf-8") as file:
        metadata_document = json.load(file)

    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("Giriş sinyali tek boyutlu ve boş olmayan bir dizi olmalıdır.")

    if not np.all(np.isfinite(samples.real)):
        raise ValueError("Giriş sinyalinin gerçek bileşeninde NaN veya Inf bulundu.")

    if not np.all(np.isfinite(samples.imag)):
        raise ValueError("Giriş sinyalinin sanal bileşeninde NaN veya Inf bulundu.")

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz pozitif olmalıdır.")

    return SignalPackage(
        samples=samples,
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
        metadata=metadata_document,
    )


def apply_carrier_impairment(
    samples: ComplexArray,
    sample_rate_hz: float,
    cfo_hz: float,
    phase_offset_degrees: float,
) -> tuple[ComplexArray, FloatArray]:
    """
    CFO ve sabit faz kayması uygular.

    y[n] = x[n] * exp(j * (2*pi*CFO*n/Fs + phi))
    """

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz pozitif olmalıdır.")

    if abs(cfo_hz) >= sample_rate_hz / 2.0:
        raise ValueError(
            "CFO büyüklüğü Nyquist frekansından küçük olmalıdır."
        )

    sample_indices = np.arange(len(samples), dtype=np.float64)
    phase_offset_radians = np.deg2rad(phase_offset_degrees)

    phase_radians = (
        2.0 * np.pi * cfo_hz * sample_indices / sample_rate_hz
        + phase_offset_radians
    )

    rotation = np.exp(1j * phase_radians)
    impaired = samples * rotation

    return (
        impaired.astype(np.complex128),
        phase_radians.astype(np.float64),
    )


def add_complex_awgn(
    samples: ComplexArray,
    target_snr_db: float,
    rng: np.random.Generator,
) -> tuple[ComplexArray, ComplexArray, float]:
    """
    Hedef SNR değerine göre sıfır ortalamalı kompleks AWGN ekler.

    Kompleks gürültünün I ve Q bileşenleri eşit varyansa sahiptir.
    """

    signal_power = float(np.mean(np.abs(samples) ** 2))

    if signal_power <= 0.0 or not np.isfinite(signal_power):
        raise ValueError("Sinyal gücü pozitif ve sonlu olmalıdır.")

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power_target = signal_power / snr_linear
    component_std = np.sqrt(noise_power_target / 2.0)

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


def power_spectrum_db(
    samples: ComplexArray,
    sample_rate_hz: float,
    fft_size: int = 16384,
) -> tuple[FloatArray, FloatArray]:
    """Normalize güç spektrumunu dB cinsinden hesaplar."""

    if fft_size <= 0:
        raise ValueError("fft_size pozitif olmalıdır.")

    segment_length = min(len(samples), fft_size)
    segment = samples[:segment_length]
    window = np.hanning(segment_length)

    spectrum = np.fft.fftshift(
        np.fft.fft(segment * window, n=fft_size)
    )
    power = np.abs(spectrum) ** 2
    maximum_power = float(np.max(power))

    if maximum_power > 0.0:
        power /= maximum_power

    spectrum_db = 10.0 * np.log10(np.maximum(power, 1e-15))
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz)
    )

    return (
        frequencies.astype(np.float64),
        spectrum_db.astype(np.float64),
    )


def create_plots(
    clean_samples: ComplexArray,
    carrier_impaired_samples: ComplexArray,
    received_samples: ComplexArray,
    sample_rate_hz: float,
    samples_per_symbol: int,
    output_path: Path,
) -> None:
    """Zaman alanı, spektrum ve örnek konstelasyon grafiklerini kaydeder."""

    figure, axes = plt.subplots(3, 1, figsize=(12, 13))

    sample_count = min(500, len(clean_samples))
    time_axis_ms = (
        np.arange(sample_count, dtype=np.float64)
        / sample_rate_hz
        * 1_000.0
    )

    axes[0].plot(
        time_axis_ms,
        clean_samples[:sample_count].real,
        label="Temiz I",
        linewidth=1.2,
    )
    axes[0].plot(
        time_axis_ms,
        received_samples[:sample_count].real,
        label="Bozulmuş I",
        linewidth=0.9,
        alpha=0.8,
    )
    axes[0].set_title("Zaman Alanı — Temiz ve Bozulmuş Sinyal")
    axes[0].set_xlabel("Zaman (ms)")
    axes[0].set_ylabel("Genlik")
    axes[0].grid(True)
    axes[0].legend()

    clean_frequency, clean_spectrum = power_spectrum_db(
        clean_samples,
        sample_rate_hz,
    )
    received_frequency, received_spectrum = power_spectrum_db(
        received_samples,
        sample_rate_hz,
    )

    axes[1].plot(
        clean_frequency / 1_000.0,
        clean_spectrum,
        label="Temiz",
    )
    axes[1].plot(
        received_frequency / 1_000.0,
        received_spectrum,
        label="CFO + Faz + AWGN",
        alpha=0.8,
    )
    axes[1].set_title("Normalize Güç Spektrumu")
    axes[1].set_xlabel("Frekans (kHz)")
    axes[1].set_ylabel("Normalize Güç (dB)")
    axes[1].set_ylim(-100.0, 5.0)
    axes[1].grid(True)
    axes[1].legend()

    symbol_center = samples_per_symbol // 2
    clean_symbol_samples = clean_samples[
        symbol_center::samples_per_symbol
    ]
    received_symbol_samples = received_samples[
        symbol_center::samples_per_symbol
    ]

    plot_count = min(
        1500,
        len(clean_symbol_samples),
        len(received_symbol_samples),
    )

    axes[2].scatter(
        clean_symbol_samples[:plot_count].real,
        clean_symbol_samples[:plot_count].imag,
        s=8,
        alpha=0.35,
        label="Temiz örnekler",
    )
    axes[2].scatter(
        received_symbol_samples[:plot_count].real,
        received_symbol_samples[:plot_count].imag,
        s=8,
        alpha=0.35,
        label="Bozulmuş örnekler",
    )
    axes[2].axhline(0.0, linewidth=0.8)
    axes[2].axvline(0.0, linewidth=0.8)
    axes[2].set_title("Örnek Konstelasyon Görünümü")
    axes[2].set_xlabel("In-Phase (I)")
    axes[2].set_ylabel("Quadrature (Q)")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    STEP03_DIRECTORY.mkdir(parents=True, exist_ok=True)

    package = load_step02_package()
    rng = np.random.default_rng(RANDOM_SEED)

    carrier_impaired, phase_radians = apply_carrier_impairment(
        samples=package.samples,
        sample_rate_hz=package.sample_rate_hz,
        cfo_hz=CFO_HZ,
        phase_offset_degrees=PHASE_OFFSET_DEGREES,
    )

    clean_power = float(np.mean(np.abs(package.samples) ** 2))
    carrier_impaired_power = float(
        np.mean(np.abs(carrier_impaired) ** 2)
    )

    if not np.isclose(
        clean_power,
        carrier_impaired_power,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise RuntimeError(
            "CFO/faz işlemi sinyal gücünü beklenmedik şekilde değiştirdi."
        )

    received, noise, achieved_snr_db = add_complex_awgn(
        samples=carrier_impaired,
        target_snr_db=TARGET_SNR_DB,
        rng=rng,
    )

    if len(received) != len(package.samples):
        raise RuntimeError("Bozunum motoru sinyal uzunluğunu değiştirdi.")

    if not np.all(np.isfinite(received.real)):
        raise RuntimeError("Alınan sinyalin gerçek kısmında NaN veya Inf var.")

    if not np.all(np.isfinite(received.imag)):
        raise RuntimeError("Alınan sinyalin sanal kısmında NaN veya Inf var.")

    snr_error_db = abs(achieved_snr_db - TARGET_SNR_DB)

    if snr_error_db > 0.25:
        raise RuntimeError(
            "Gerçekleşen SNR hedef değerden beklenenden fazla sapıyor."
        )

    noise_nmse = calculate_nmse(
        reference=carrier_impaired,
        estimate=received,
    )

    np.save(
        STEP03_DIRECTORY / "clean_samples.npy",
        package.samples,
    )
    np.save(
        STEP03_DIRECTORY / "carrier_impaired_samples.npy",
        carrier_impaired,
    )
    np.save(
        STEP03_DIRECTORY / "received_samples.npy",
        received,
    )
    np.save(
        STEP03_DIRECTORY / "noise_samples.npy",
        noise,
    )
    np.save(
        STEP03_DIRECTORY / "phase_trajectory_radians.npy",
        phase_radians,
    )
    np.save(
        STEP03_DIRECTORY / "bits.npy",
        package.bits,
    )
    np.save(
        STEP03_DIRECTORY / "symbols.npy",
        package.symbols,
    )

    np.savez_compressed(
        STEP03_DIRECTORY / "impaired_signal_record.npz",
        clean_samples=package.samples,
        carrier_impaired_samples=carrier_impaired,
        received_samples=received,
        noise_samples=noise,
        phase_trajectory_radians=phase_radians,
        bits=package.bits,
        symbols=package.symbols,
        sample_rate_hz=np.float64(package.sample_rate_hz),
        symbol_rate_hz=np.float64(package.symbol_rate_hz),
        samples_per_symbol=np.int64(package.samples_per_symbol),
        cfo_hz=np.float64(CFO_HZ),
        phase_offset_degrees=np.float64(PHASE_OFFSET_DEGREES),
        target_snr_db=np.float64(TARGET_SNR_DB),
        achieved_snr_db=np.float64(achieved_snr_db),
    )

    metadata = {
        "project": "GENESIS-DSP",
        "step": 3,
        "description": "CFO, phase offset and complex AWGN impairment engine",
        "random_seed": RANDOM_SEED,
        "input_step": 2,
        "number_of_samples": int(len(package.samples)),
        "sample_rate_hz": package.sample_rate_hz,
        "symbol_rate_hz": package.symbol_rate_hz,
        "samples_per_symbol": package.samples_per_symbol,
        "impairments": {
            "carrier_frequency_offset_hz": CFO_HZ,
            "phase_offset_degrees": PHASE_OFFSET_DEGREES,
            "target_snr_db": TARGET_SNR_DB,
            "achieved_snr_db": achieved_snr_db,
            "snr_error_db": snr_error_db,
        },
        "measurements": {
            "clean_power": clean_power,
            "carrier_impaired_power": carrier_impaired_power,
            "noise_power": float(np.mean(np.abs(noise) ** 2)),
            "received_power": float(np.mean(np.abs(received) ** 2)),
            "noise_nmse": noise_nmse,
        },
        "validations": {
            "sample_length_preserved": True,
            "carrier_rotation_power_preserved": True,
            "finite_output": True,
            "snr_tolerance_passed": True,
        },
    }

    with (STEP03_DIRECTORY / "metadata.json").open(
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
        clean_samples=package.samples,
        carrier_impaired_samples=carrier_impaired,
        received_samples=received,
        sample_rate_hz=package.sample_rate_hz,
        samples_per_symbol=package.samples_per_symbol,
        output_path=STEP03_DIRECTORY / "impairments_overview.png",
    )

    print()
    print("=" * 68)
    print("GENESIS-DSP — ADIM 03 BAŞARIYLA TAMAMLANDI")
    print("=" * 68)
    print(f"Örnek sayısı                 : {len(package.samples)}")
    print(f"CFO                           : {CFO_HZ:,.3f} Hz")
    print(f"Faz kayması                  : {PHASE_OFFSET_DEGREES:.3f} derece")
    print(f"Hedef SNR                    : {TARGET_SNR_DB:.6f} dB")
    print(f"Gerçekleşen SNR              : {achieved_snr_db:.6f} dB")
    print(f"SNR hatası                   : {snr_error_db:.6f} dB")
    print(f"Temiz sinyal gücü            : {clean_power:.12f}")
    print(f"CFO sonrası sinyal gücü      : {carrier_impaired_power:.12f}")
    print(f"Gürültü NMSE                 : {noise_nmse:.12e}")
    print(f"Sonuç klasörü                : {STEP03_DIRECTORY}")
    print("=" * 68)


if __name__ == "__main__":
    main()
