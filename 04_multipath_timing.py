"""
GENESIS-DSP — Adım 04
Çok yollu kanal ve kesirli zamanlama kayması motoru.

Bu program:
1. Adım 03 paketini yükler.
2. CFO ve faz kayması uygulanmış sinyali çok yollu kanaldan geçirir.
3. Kesirli örnek zamanlama kayması uygular.
4. Kanal sonrasında hedef SNR seviyesinde yeni kompleks AWGN ekler.
5. Kanal, timing ve SNR doğrulamalarını yapar.
6. Tüm ground-truth verilerini, metadata'yı ve grafikleri kaydeder.

Çalıştırma:
    python 04_multipath_timing.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP03_DIRECTORY = BASE_DIRECTORY / "outputs" / "step03"
STEP04_DIRECTORY = BASE_DIRECTORY / "outputs" / "step04"

INPUT_PACKAGE = STEP03_DIRECTORY / "impaired_signal_record.npz"
INPUT_METADATA = STEP03_DIRECTORY / "metadata.json"

RANDOM_SEED = 20260716 + 4

# Çok yollu kanal tanımı
PATH_DELAYS_SAMPLES = (0, 3, 8)
PATH_MAGNITUDES = (1.0, 0.48, 0.23)
PATH_PHASES_DEGREES = (0.0, 42.0, -71.0)

# Pozitif değer, sinyali bu miktarda geciktirir.
FRACTIONAL_DELAY_SAMPLES = 0.37
FRACTIONAL_DELAY_FILTER_TAPS = 41

# Adım 03'teki hedef SNR değeri korunur.
DEFAULT_TARGET_SNR_DB = 12.0

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Step03Package:
    clean_samples: ComplexArray
    carrier_impaired_samples: ComplexArray
    bits: NDArray[np.int8]
    symbols: ComplexArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    cfo_hz: float
    phase_offset_degrees: float
    target_snr_db: float
    metadata: dict[str, Any]


def load_step03_package() -> Step03Package:
    """Adım 03 veri paketini ve metadata dosyasını yükler."""

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 03 veri paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python 03_basic_impairments.py"
        )

    if not INPUT_METADATA.exists():
        raise FileNotFoundError(
            f"Adım 03 metadata dosyası bulunamadı: {INPUT_METADATA}"
        )

    with np.load(INPUT_PACKAGE, allow_pickle=False) as package:
        clean_samples = package["clean_samples"].astype(np.complex128)
        carrier_impaired_samples = package[
            "carrier_impaired_samples"
        ].astype(np.complex128)
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        sample_rate_hz = float(package["sample_rate_hz"])
        symbol_rate_hz = float(package["symbol_rate_hz"])
        samples_per_symbol = int(package["samples_per_symbol"])
        cfo_hz = float(package["cfo_hz"])
        phase_offset_degrees = float(package["phase_offset_degrees"])

        if "target_snr_db" in package.files:
            target_snr_db = float(package["target_snr_db"])
        else:
            target_snr_db = DEFAULT_TARGET_SNR_DB

    with INPUT_METADATA.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    arrays_to_check = {
        "clean_samples": clean_samples,
        "carrier_impaired_samples": carrier_impaired_samples,
        "symbols": symbols,
    }

    for name, array in arrays_to_check.items():
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

    return Step03Package(
        clean_samples=clean_samples,
        carrier_impaired_samples=carrier_impaired_samples,
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
        cfo_hz=cfo_hz,
        phase_offset_degrees=phase_offset_degrees,
        target_snr_db=target_snr_db,
        metadata=metadata,
    )


def build_multipath_channel(
    delays_samples: tuple[int, ...],
    magnitudes: tuple[float, ...],
    phases_degrees: tuple[float, ...],
) -> ComplexArray:
    """
    Seyrek kompleks FIR çok yollu kanal dürtü cevabı oluşturur.

    Kanal katsayıları toplam enerji 1 olacak şekilde normalize edilir.
    """

    if not (
        len(delays_samples)
        == len(magnitudes)
        == len(phases_degrees)
    ):
        raise ValueError(
            "Gecikme, büyüklük ve faz listeleri aynı uzunlukta olmalıdır."
        )

    if len(delays_samples) == 0:
        raise ValueError("En az bir kanal yolu bulunmalıdır.")

    if any(delay < 0 for delay in delays_samples):
        raise ValueError("Kanal yolu gecikmeleri negatif olamaz.")

    if any(magnitude < 0.0 for magnitude in magnitudes):
        raise ValueError("Kanal yolu büyüklükleri negatif olamaz.")

    if len(set(delays_samples)) != len(delays_samples):
        raise ValueError("Kanal yolu gecikmeleri benzersiz olmalıdır.")

    channel = np.zeros(
        max(delays_samples) + 1,
        dtype=np.complex128,
    )

    for delay, magnitude, phase_degrees in zip(
        delays_samples,
        magnitudes,
        phases_degrees,
        strict=True,
    ):
        phase_radians = np.deg2rad(phase_degrees)
        channel[delay] = magnitude * np.exp(1j * phase_radians)

    channel_energy = float(np.sum(np.abs(channel) ** 2))

    if channel_energy <= 0.0 or not np.isfinite(channel_energy):
        raise RuntimeError("Geçerli bir çok yollu kanal oluşturulamadı.")

    channel /= np.sqrt(channel_energy)

    return channel.astype(np.complex128)


def apply_multipath(
    samples: ComplexArray,
    channel_taps: ComplexArray,
) -> ComplexArray:
    """Sinyali kompleks FIR çok yollu kanaldan geçirir."""

    if samples.ndim != 1 or channel_taps.ndim != 1:
        raise ValueError("samples ve channel_taps tek boyutlu olmalıdır.")

    output = np.convolve(
        samples,
        channel_taps,
        mode="full",
    )

    return output.astype(np.complex128)


def design_fractional_delay_filter(
    delay_samples: float,
    number_of_taps: int,
) -> FloatArray:
    """
    Pencerelenmiş sinc yöntemiyle kesirli gecikme FIR filtresi tasarlar.

    number_of_taps tek sayı olmalıdır. Filtrenin toplam gecikmesi:
        (number_of_taps - 1) / 2 + delay_samples
    """

    if not np.isfinite(delay_samples):
        raise ValueError("delay_samples sonlu olmalıdır.")

    if not -0.5 < delay_samples < 0.5:
        raise ValueError(
            "Bu aşamada fractional delay -0.5 ile +0.5 arasında olmalıdır."
        )

    if number_of_taps < 5 or number_of_taps % 2 == 0:
        raise ValueError(
            "number_of_taps en az 5 ve tek sayı olmalıdır."
        )

    half = (number_of_taps - 1) // 2
    indices = np.arange(-half, half + 1, dtype=np.float64)

    taps = np.sinc(indices - delay_samples)
    taps *= np.hamming(number_of_taps)

    dc_gain = float(np.sum(taps))

    if np.isclose(dc_gain, 0.0, atol=1e-15):
        raise RuntimeError("Fractional-delay filtresinin DC kazancı sıfır.")

    taps /= dc_gain

    if not np.all(np.isfinite(taps)):
        raise RuntimeError("Fractional-delay filtresinde NaN veya Inf oluştu.")

    return taps.astype(np.float64)


def apply_fractional_delay(
    samples: ComplexArray,
    filter_taps: FloatArray,
) -> ComplexArray:
    """
    Kesirli gecikme filtresini uygular.

    'full' konvolüsyon kullanılır; böylece filtre geçişleri ve gecikme
    ground-truth içinde açıkça korunur.
    """

    output = np.convolve(
        samples,
        filter_taps,
        mode="full",
    )

    return output.astype(np.complex128)


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


def estimate_filter_delay_samples(filter_taps: FloatArray) -> float:
    """
    Filtre enerji merkezinden etkili gecikmeyi tahmin eder.
    Bu değer doğrulama amacıyla kullanılır.
    """

    energy = np.abs(filter_taps) ** 2
    total_energy = float(np.sum(energy))

    if total_energy <= 0.0:
        raise ValueError("Filtre enerjisi pozitif olmalıdır.")

    indices = np.arange(len(filter_taps), dtype=np.float64)

    return float(np.sum(indices * energy) / total_energy)


def calculate_frequency_response(
    taps: ComplexArray | FloatArray,
    sample_rate_hz: float,
    fft_size: int = 4096,
) -> tuple[FloatArray, FloatArray]:
    """FIR filtrenin normalize genlik frekans cevabını hesaplar."""

    response = np.fft.fftshift(np.fft.fft(taps, n=fft_size))
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz)
    )

    magnitude = np.abs(response)
    maximum = float(np.max(magnitude))

    if maximum > 0.0:
        magnitude /= maximum

    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude, 1e-12)
    )

    return (
        frequencies.astype(np.float64),
        magnitude_db.astype(np.float64),
    )


def create_plots(
    channel_taps: ComplexArray,
    fractional_delay_taps: FloatArray,
    clean_samples: ComplexArray,
    noiseless_received: ComplexArray,
    received_samples: ComplexArray,
    sample_rate_hz: float,
    output_path: Path,
) -> None:
    """Kanal, timing filtresi, zaman alanı ve spektrum grafiklerini üretir."""

    figure, axes = plt.subplots(4, 1, figsize=(12, 16))

    channel_indices = np.arange(len(channel_taps))
    axes[0].stem(
        channel_indices,
        np.abs(channel_taps),
    )
    axes[0].set_title("Çok Yollu Kanal Dürtü Cevabı — |h[n]|")
    axes[0].set_xlabel("Gecikme (örnek)")
    axes[0].set_ylabel("Büyüklük")
    axes[0].grid(True)

    delay_indices = np.arange(len(fractional_delay_taps))
    axes[1].stem(
        delay_indices,
        fractional_delay_taps,
    )
    axes[1].set_title("Kesirli Gecikme FIR Filtresi")
    axes[1].set_xlabel("Tap indeksi")
    axes[1].set_ylabel("Katsayı")
    axes[1].grid(True)

    plot_count = min(
        500,
        len(clean_samples),
        len(received_samples),
    )

    time_axis_ms = (
        np.arange(plot_count, dtype=np.float64)
        / sample_rate_hz
        * 1_000.0
    )

    axes[2].plot(
        time_axis_ms,
        clean_samples[:plot_count].real,
        label="Temiz I",
        linewidth=1.2,
    )
    axes[2].plot(
        time_axis_ms,
        received_samples[:plot_count].real,
        label="Kanal + Timing + AWGN",
        linewidth=0.9,
        alpha=0.8,
    )
    axes[2].set_title("Zaman Alanı Karşılaştırması")
    axes[2].set_xlabel("Zaman (ms)")
    axes[2].set_ylabel("Genlik")
    axes[2].grid(True)
    axes[2].legend()

    channel_frequency, channel_response_db = calculate_frequency_response(
        channel_taps,
        sample_rate_hz,
    )
    delay_frequency, delay_response_db = calculate_frequency_response(
        fractional_delay_taps,
        sample_rate_hz,
    )

    axes[3].plot(
        channel_frequency / 1_000.0,
        channel_response_db,
        label="Çok yollu kanal",
    )
    axes[3].plot(
        delay_frequency / 1_000.0,
        delay_response_db,
        label="Fractional-delay filtresi",
        alpha=0.85,
    )
    axes[3].set_title("Filtre Frekans Cevapları")
    axes[3].set_xlabel("Frekans (kHz)")
    axes[3].set_ylabel("Normalize Genlik (dB)")
    axes[3].set_ylim(-60.0, 5.0)
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    STEP04_DIRECTORY.mkdir(parents=True, exist_ok=True)

    package = load_step03_package()
    rng = np.random.default_rng(RANDOM_SEED)

    channel_taps = build_multipath_channel(
        delays_samples=PATH_DELAYS_SAMPLES,
        magnitudes=PATH_MAGNITUDES,
        phases_degrees=PATH_PHASES_DEGREES,
    )

    channel_output = apply_multipath(
        samples=package.carrier_impaired_samples,
        channel_taps=channel_taps,
    )

    fractional_delay_taps = design_fractional_delay_filter(
        delay_samples=FRACTIONAL_DELAY_SAMPLES,
        number_of_taps=FRACTIONAL_DELAY_FILTER_TAPS,
    )

    noiseless_received = apply_fractional_delay(
        samples=channel_output,
        filter_taps=fractional_delay_taps,
    )

    received_samples, noise_samples, achieved_snr_db = add_complex_awgn(
        samples=noiseless_received,
        target_snr_db=package.target_snr_db,
        rng=rng,
    )

    expected_channel_length = (
        len(package.carrier_impaired_samples)
        + len(channel_taps)
        - 1
    )

    expected_output_length = (
        expected_channel_length
        + len(fractional_delay_taps)
        - 1
    )

    if len(channel_output) != expected_channel_length:
        raise RuntimeError("Çok yollu kanal çıkış uzunluğu hatalı.")

    if len(received_samples) != expected_output_length:
        raise RuntimeError("Timing filtresi çıkış uzunluğu hatalı.")

    channel_energy = float(np.sum(np.abs(channel_taps) ** 2))
    fractional_delay_dc_gain = float(np.sum(fractional_delay_taps))

    if not np.isclose(channel_energy, 1.0, atol=1e-12):
        raise RuntimeError("Çok yollu kanal birim enerjili değil.")

    if not np.isclose(fractional_delay_dc_gain, 1.0, atol=1e-12):
        raise RuntimeError("Fractional-delay filtresi birim DC kazançlı değil.")

    if not np.all(np.isfinite(received_samples.real)):
        raise RuntimeError("Çıkışın gerçek kısmında NaN veya Inf bulundu.")

    if not np.all(np.isfinite(received_samples.imag)):
        raise RuntimeError("Çıkışın sanal kısmında NaN veya Inf bulundu.")

    snr_error_db = abs(achieved_snr_db - package.target_snr_db)

    if snr_error_db > 0.25:
        raise RuntimeError("Gerçekleşen SNR tolerans dışında.")

    estimated_filter_delay = estimate_filter_delay_samples(
        fractional_delay_taps
    )
    nominal_integer_delay = (len(fractional_delay_taps) - 1) / 2.0
    estimated_fractional_component = (
        estimated_filter_delay - nominal_integer_delay
    )

    np.save(
        STEP04_DIRECTORY / "channel_taps.npy",
        channel_taps,
    )
    np.save(
        STEP04_DIRECTORY / "fractional_delay_taps.npy",
        fractional_delay_taps,
    )
    np.save(
        STEP04_DIRECTORY / "channel_output.npy",
        channel_output,
    )
    np.save(
        STEP04_DIRECTORY / "noiseless_received_samples.npy",
        noiseless_received,
    )
    np.save(
        STEP04_DIRECTORY / "received_samples.npy",
        received_samples,
    )
    np.save(
        STEP04_DIRECTORY / "noise_samples.npy",
        noise_samples,
    )
    np.save(
        STEP04_DIRECTORY / "bits.npy",
        package.bits,
    )
    np.save(
        STEP04_DIRECTORY / "symbols.npy",
        package.symbols,
    )

    np.savez_compressed(
        STEP04_DIRECTORY / "channel_timing_record.npz",
        clean_samples=package.clean_samples,
        carrier_impaired_samples=package.carrier_impaired_samples,
        channel_output=channel_output,
        noiseless_received_samples=noiseless_received,
        received_samples=received_samples,
        noise_samples=noise_samples,
        channel_taps=channel_taps,
        fractional_delay_taps=fractional_delay_taps,
        bits=package.bits,
        symbols=package.symbols,
        sample_rate_hz=np.float64(package.sample_rate_hz),
        symbol_rate_hz=np.float64(package.symbol_rate_hz),
        samples_per_symbol=np.int64(package.samples_per_symbol),
        cfo_hz=np.float64(package.cfo_hz),
        phase_offset_degrees=np.float64(
            package.phase_offset_degrees
        ),
        path_delays_samples=np.asarray(
            PATH_DELAYS_SAMPLES,
            dtype=np.int64,
        ),
        path_magnitudes=np.asarray(
            PATH_MAGNITUDES,
            dtype=np.float64,
        ),
        path_phases_degrees=np.asarray(
            PATH_PHASES_DEGREES,
            dtype=np.float64,
        ),
        fractional_delay_samples=np.float64(
            FRACTIONAL_DELAY_SAMPLES
        ),
        target_snr_db=np.float64(package.target_snr_db),
        achieved_snr_db=np.float64(achieved_snr_db),
    )

    metadata = {
        "project": "GENESIS-DSP",
        "step": 4,
        "description": (
            "Multipath channel and fractional timing offset engine"
        ),
        "random_seed": RANDOM_SEED,
        "input_step": 3,
        "signal": {
            "sample_rate_hz": package.sample_rate_hz,
            "symbol_rate_hz": package.symbol_rate_hz,
            "samples_per_symbol": package.samples_per_symbol,
            "input_number_of_samples": int(
                len(package.carrier_impaired_samples)
            ),
            "output_number_of_samples": int(len(received_samples)),
        },
        "existing_impairments": {
            "cfo_hz": package.cfo_hz,
            "phase_offset_degrees": package.phase_offset_degrees,
        },
        "multipath_channel": {
            "path_delays_samples": list(PATH_DELAYS_SAMPLES),
            "path_magnitudes_before_normalization": list(
                PATH_MAGNITUDES
            ),
            "path_phases_degrees": list(PATH_PHASES_DEGREES),
            "normalized_complex_taps_real": [
                float(value.real) for value in channel_taps
            ],
            "normalized_complex_taps_imag": [
                float(value.imag) for value in channel_taps
            ],
            "channel_energy": channel_energy,
        },
        "fractional_timing": {
            "requested_fractional_delay_samples": (
                FRACTIONAL_DELAY_SAMPLES
            ),
            "filter_number_of_taps": FRACTIONAL_DELAY_FILTER_TAPS,
            "filter_integer_group_delay_samples": (
                nominal_integer_delay
            ),
            "estimated_total_filter_delay_samples": (
                estimated_filter_delay
            ),
            "estimated_fractional_component_samples": (
                estimated_fractional_component
            ),
            "filter_dc_gain": fractional_delay_dc_gain,
        },
        "awgn": {
            "target_snr_db": package.target_snr_db,
            "achieved_snr_db": achieved_snr_db,
            "snr_error_db": snr_error_db,
            "signal_power_before_noise": float(
                np.mean(np.abs(noiseless_received) ** 2)
            ),
            "noise_power": float(
                np.mean(np.abs(noise_samples) ** 2)
            ),
        },
        "processing_order": [
            "clean QPSK with RRC pulse shaping",
            "carrier frequency offset",
            "constant phase offset",
            "multipath FIR channel",
            "fractional timing delay",
            "complex AWGN",
        ],
        "validations": {
            "channel_energy_normalized": True,
            "fractional_delay_dc_gain_normalized": True,
            "output_length_verified": True,
            "finite_output": True,
            "snr_tolerance_passed": True,
        },
    }

    with (STEP04_DIRECTORY / "metadata.json").open(
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
        channel_taps=channel_taps,
        fractional_delay_taps=fractional_delay_taps,
        clean_samples=package.clean_samples,
        noiseless_received=noiseless_received,
        received_samples=received_samples,
        sample_rate_hz=package.sample_rate_hz,
        output_path=STEP04_DIRECTORY / "channel_timing_overview.png",
    )

    print()
    print("=" * 72)
    print("GENESIS-DSP — ADIM 04 BAŞARIYLA TAMAMLANDI")
    print("=" * 72)
    print(f"Giriş örnek sayısı             : {len(package.carrier_impaired_samples)}")
    print(f"Kanal tap sayısı               : {len(channel_taps)}")
    print(f"Kanal yolu gecikmeleri         : {PATH_DELAYS_SAMPLES}")
    print(f"Kanal enerjisi                 : {channel_energy:.12f}")
    print(f"Kesirli timing kayması         : {FRACTIONAL_DELAY_SAMPLES:.6f} örnek")
    print(f"Timing filtresi tap sayısı     : {len(fractional_delay_taps)}")
    print(f"Timing filtresi DC kazancı     : {fractional_delay_dc_gain:.12f}")
    print(f"Çıkış örnek sayısı             : {len(received_samples)}")
    print(f"Hedef SNR                      : {package.target_snr_db:.6f} dB")
    print(f"Gerçekleşen SNR                : {achieved_snr_db:.6f} dB")
    print(f"SNR hatası                     : {snr_error_db:.6f} dB")
    print(f"Sonuç klasörü                  : {STEP04_DIRECTORY}")
    print("=" * 72)


if __name__ == "__main__":
    main()
