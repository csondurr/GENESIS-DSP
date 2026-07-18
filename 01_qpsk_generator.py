

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


SEED = 20260716
NUM_SYMBOLS = 4096
SAMPLES_PER_SYMBOL = 4
SAMPLE_RATE_HZ = 1_000_000.0
SYMBOL_RATE_HZ = SAMPLE_RATE_HZ / SAMPLES_PER_SYMBOL
RRC_ROLL_OFF = 0.35
RRC_SPAN_SYMBOLS = 10

BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIRECTORY / "outputs" / "step01"

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.int8]


def root_raised_cosine(
    roll_off: float,
    samples_per_symbol: int,
    span_symbols: int,
) -> FloatArray:
    """Birim enerjili Root Raised Cosine FIR filtresi üretir."""

    if not 0.0 < roll_off <= 1.0:
        raise ValueError("roll_off değeri 0 ile 1 arasında olmalıdır.")
    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol en az 2 olmalıdır.")
    if span_symbols < 2 or span_symbols % 2 != 0:
        raise ValueError("span_symbols en az 2 ve çift olmalıdır.")

    half_length = span_symbols * samples_per_symbol // 2
    indices = np.arange(-half_length, half_length + 1, dtype=np.float64)
    time = indices / samples_per_symbol
    taps = np.zeros_like(time)
    beta = roll_off

    for index, t_value in enumerate(time):
        if np.isclose(t_value, 0.0, atol=1e-12):
            taps[index] = 1.0 + beta * ((4.0 / np.pi) - 1.0)
        elif np.isclose(abs(t_value), 1.0 / (4.0 * beta), atol=1e-12):
            taps[index] = (
                beta
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi)
                    * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi)
                    * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * t_value * (1.0 - beta))
                + 4.0
                * beta
                * t_value
                * np.cos(np.pi * t_value * (1.0 + beta))
            )
            denominator = (
                np.pi
                * t_value
                * (1.0 - (4.0 * beta * t_value) ** 2)
            )
            taps[index] = numerator / denominator

    energy = float(np.sum(np.abs(taps) ** 2))
    if energy <= 0.0 or not np.isfinite(energy):
        raise RuntimeError("Geçerli bir RRC filtresi üretilemedi.")

    taps /= np.sqrt(energy)
    return taps.astype(np.float64)


def generate_qpsk_symbols(
    number_of_symbols: int,
    rng: np.random.Generator,
) -> tuple[BitArray, ComplexArray]:
    """Rastgele bitlerden birim güçlü QPSK sembolleri üretir."""

    if number_of_symbols <= 0:
        raise ValueError("number_of_symbols pozitif olmalıdır.")

    bits = rng.integers(
        low=0,
        high=2,
        size=(number_of_symbols, 2),
        dtype=np.int8,
    )

    in_phase = 1.0 - 2.0 * bits[:, 0].astype(np.float64)
    quadrature = 1.0 - 2.0 * bits[:, 1].astype(np.float64)
    symbols = (in_phase + 1j * quadrature) / np.sqrt(2.0)

    return bits, symbols.astype(np.complex128)


def pulse_shape_signal(
    symbols: ComplexArray,
    samples_per_symbol: int,
    filter_taps: FloatArray,
) -> ComplexArray:
    """Sembolleri yükseltir ve RRC filtresinden geçirir."""

    upsampled = np.zeros(
        len(symbols) * samples_per_symbol,
        dtype=np.complex128,
    )
    upsampled[::samples_per_symbol] = symbols

    return np.convolve(
        upsampled,
        filter_taps,
        mode="full",
    ).astype(np.complex128)


def create_overview_plot(
    symbols: ComplexArray,
    transmit_signal: ComplexArray,
    sample_rate_hz: float,
    output_path: Path,
) -> None:
    """Konstelasyon, zaman alanı ve spektrum grafiklerini kaydeder."""

    figure, axes = plt.subplots(3, 1, figsize=(12, 12))

    symbol_count = min(800, len(symbols))
    axes[0].scatter(
        symbols[:symbol_count].real,
        symbols[:symbol_count].imag,
        s=16,
        alpha=0.55,
    )
    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].axvline(0.0, linewidth=0.8)
    axes[0].set_title("Temiz QPSK Konstelasyonu")
    axes[0].set_xlabel("In-Phase (I)")
    axes[0].set_ylabel("Quadrature (Q)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True)

    sample_count = min(400, len(transmit_signal))
    time_axis_ms = (
        np.arange(sample_count, dtype=np.float64)
        / sample_rate_hz
        * 1_000.0
    )
    axes[1].plot(
        time_axis_ms,
        transmit_signal[:sample_count].real,
        label="I",
    )
    axes[1].plot(
        time_axis_ms,
        transmit_signal[:sample_count].imag,
        label="Q",
    )
    axes[1].set_title("Pulse-Shaped QPSK — Zaman Alanı")
    axes[1].set_xlabel("Zaman (ms)")
    axes[1].set_ylabel("Genlik")
    axes[1].grid(True)
    axes[1].legend()

    fft_size = 8192
    segment_length = min(fft_size, len(transmit_signal))
    segment = transmit_signal[:segment_length]
    windowed = segment * np.hanning(segment_length)

    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    power = np.abs(spectrum) ** 2
    maximum_power = float(np.max(power))

    if maximum_power > 0.0:
        power /= maximum_power

    power_db = 10.0 * np.log10(np.maximum(power, 1e-14))
    frequency_axis_khz = (
        np.fft.fftshift(
            np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz)
        )
        / 1_000.0
    )

    axes[2].plot(frequency_axis_khz, power_db)
    axes[2].set_title("Normalize Güç Spektrumu")
    axes[2].set_xlabel("Frekans (kHz)")
    axes[2].set_ylabel("Normalize Güç (dB)")
    axes[2].set_ylim(-100.0, 5.0)
    axes[2].grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    bits, symbols = generate_qpsk_symbols(
        number_of_symbols=NUM_SYMBOLS,
        rng=rng,
    )

    rrc_taps = root_raised_cosine(
        roll_off=RRC_ROLL_OFF,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        span_symbols=RRC_SPAN_SYMBOLS,
    )

    transmit_signal = pulse_shape_signal(
        symbols=symbols,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        filter_taps=rrc_taps,
    )

    if not np.all(np.isfinite(symbols)):
        raise RuntimeError("QPSK sembollerinde NaN veya Inf bulundu.")
    if not np.all(np.isfinite(transmit_signal)):
        raise RuntimeError("Çıkış sinyalinde NaN veya Inf bulundu.")

    average_symbol_power = float(np.mean(np.abs(symbols) ** 2))
    filter_energy = float(np.sum(np.abs(rrc_taps) ** 2))

    if not np.isclose(average_symbol_power, 1.0, atol=1e-12):
        raise RuntimeError("QPSK sembolleri birim güce normalize edilemedi.")
    if not np.isclose(filter_energy, 1.0, atol=1e-12):
        raise RuntimeError("RRC filtresi birim enerjili değil.")

    np.save(OUTPUT_DIRECTORY / "bits.npy", bits)
    np.save(OUTPUT_DIRECTORY / "symbols.npy", symbols)
    np.save(OUTPUT_DIRECTORY / "rrc_filter_taps.npy", rrc_taps)
    np.save(OUTPUT_DIRECTORY / "clean_qpsk_signal.npy", transmit_signal)

    group_delay_samples = (len(rrc_taps) - 1) // 2

    metadata = {
        "project": "GENESIS-DSP",
        "step": 1,
        "description": "Clean pulse-shaped QPSK signal",
        "seed": SEED,
        "number_of_symbols": NUM_SYMBOLS,
        "samples_per_symbol": SAMPLES_PER_SYMBOL,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "symbol_rate_hz": SYMBOL_RATE_HZ,
        "rrc_roll_off": RRC_ROLL_OFF,
        "rrc_span_symbols": RRC_SPAN_SYMBOLS,
        "rrc_number_of_taps": int(len(rrc_taps)),
        "rrc_group_delay_samples": int(group_delay_samples),
        "number_of_output_samples": int(len(transmit_signal)),
        "average_symbol_power": average_symbol_power,
        "rrc_filter_energy": filter_energy,
        "data_type": str(transmit_signal.dtype),
    }

    with (OUTPUT_DIRECTORY / "metadata.json").open(
        mode="w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
            ensure_ascii=False,
        )

    create_overview_plot(
        symbols=symbols,
        transmit_signal=transmit_signal,
        sample_rate_hz=SAMPLE_RATE_HZ,
        output_path=OUTPUT_DIRECTORY / "qpsk_overview.png",
    )

    print()
    print("=" * 62)
    print("GENESIS-DSP — ADIM 01 BAŞARIYLA TAMAMLANDI")
    print("=" * 62)
    print(f"Üretilen sembol sayısı : {NUM_SYMBOLS}")
    print(f"Çıkış örnek sayısı     : {len(transmit_signal)}")
    print(f"Sembol gücü            : {average_symbol_power:.12f}")
    print(f"RRC filtre enerjisi    : {filter_energy:.12f}")
    print(f"Örnekleme frekansı     : {SAMPLE_RATE_HZ:,.0f} Hz")
    print(f"Sembol frekansı        : {SYMBOL_RATE_HZ:,.0f} baud")
    print(f"Sonuç klasörü          : {OUTPUT_DIRECTORY}")
    print("=" * 62)


if __name__ == "__main__":
    main()
