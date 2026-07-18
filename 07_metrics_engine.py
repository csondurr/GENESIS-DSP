"""
GENESIS-DSP — Adım 07
Merkezi performans metrikleri motoru.

Bu program:
1. Adım 06 birleşik bozunum veri paketini yükler.
2. SNR, NMSE, EVM, BER, PAPR ve güç metriklerini hesaplar.
3. QPSK hard-decision demodülatörüyle kurtarma öncesi BER'i ölçer.
4. Bilinen CFO/faz ile oracle ölçümü yaparak metrik altyapısını doğrular.
5. Kontrollü AWGN öz testi çalıştırır.
6. JSON, CSV ve grafik raporu üretir.

Çalıştırma:
    python 07_metrics_engine.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP01_METADATA = BASE_DIRECTORY / "outputs" / "step01" / "metadata.json"
STEP02_METADATA = (
    BASE_DIRECTORY
    / "outputs"
    / "step02"
    / "signal_record_metadata.json"
)
STEP06_DIRECTORY = BASE_DIRECTORY / "outputs" / "step06"
STEP07_DIRECTORY = BASE_DIRECTORY / "outputs" / "step07"

INPUT_PACKAGE = STEP06_DIRECTORY / "dataset_case_0001.npz"
INPUT_CONFIG = STEP06_DIRECTORY / "pipeline_config.json"
INPUT_METADATA = STEP06_DIRECTORY / "metadata.json"

SELF_TEST_SEED = 20260723
SELF_TEST_SNR_DB = 18.0

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.int8]


@dataclass(frozen=True)
class DatasetCase:
    clean_samples: ComplexArray
    clipped_samples: ComplexArray
    received_samples: ComplexArray
    noise_samples: ComplexArray
    channel_taps: ComplexArray
    fractional_delay_taps: FloatArray
    bits: BitArray
    symbols: ComplexArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    cfo_hz: float
    phase_offset_degrees: float
    target_snr_db: float
    achieved_snr_db: float
    config: dict[str, Any]
    metadata: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON dosyası bulunamadı: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset_case() -> DatasetCase:
    """Adım 06 veri paketini yükler ve temel bütünlük kontrolü yapar."""

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 06 veri paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python 06_impairment_pipeline.py"
        )

    config_document = load_json(INPUT_CONFIG)
    metadata = load_json(INPUT_METADATA)
    config = config_document["config"]

    with np.load(INPUT_PACKAGE, allow_pickle=False) as package:
        clean_samples = package["clean_samples"].astype(np.complex128)
        clipped_samples = package["clipped_samples"].astype(np.complex128)
        received_samples = package["received_samples"].astype(np.complex128)
        noise_samples = package["noise_samples"].astype(np.complex128)
        channel_taps = package["channel_taps"].astype(np.complex128)
        fractional_delay_taps = package[
            "fractional_delay_taps"
        ].astype(np.float64)
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        sample_rate_hz = float(package["sample_rate_hz"])
        symbol_rate_hz = float(package["symbol_rate_hz"])
        samples_per_symbol = int(package["samples_per_symbol"])
        achieved_snr_db = float(package["achieved_snr_db"])

    cfo_hz = float(config["cfo_hz"])
    phase_offset_degrees = float(config["phase_offset_degrees"])
    target_snr_db = float(config["target_snr_db"])

    complex_arrays = {
        "clean_samples": clean_samples,
        "clipped_samples": clipped_samples,
        "received_samples": received_samples,
        "noise_samples": noise_samples,
        "channel_taps": channel_taps,
        "symbols": symbols,
    }

    for name, array in complex_arrays.items():
        if array.ndim != 1 or len(array) == 0:
            raise ValueError(f"{name} tek boyutlu ve boş olmayan olmalıdır.")

        if not np.all(np.isfinite(array.real)):
            raise ValueError(f"{name} gerçek kısmında NaN veya Inf bulundu.")

        if not np.all(np.isfinite(array.imag)):
            raise ValueError(f"{name} sanal kısmında NaN veya Inf bulundu.")

    if clipped_samples.shape != received_samples.shape:
        raise ValueError(
            "clipped_samples ve received_samples aynı boyutta olmalıdır."
        )

    if received_samples.shape != noise_samples.shape:
        raise ValueError(
            "received_samples ve noise_samples aynı boyutta olmalıdır."
        )

    if bits.ndim != 2 or bits.shape[1] != 2:
        raise ValueError("bits dizisi (sembol_sayısı, 2) biçiminde olmalıdır.")

    if len(bits) != len(symbols):
        raise ValueError("Bit çifti sayısı ile sembol sayısı eşit olmalıdır.")

    return DatasetCase(
        clean_samples=clean_samples,
        clipped_samples=clipped_samples,
        received_samples=received_samples,
        noise_samples=noise_samples,
        channel_taps=channel_taps,
        fractional_delay_taps=fractional_delay_taps,
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
        cfo_hz=cfo_hz,
        phase_offset_degrees=phase_offset_degrees,
        target_snr_db=target_snr_db,
        achieved_snr_db=achieved_snr_db,
        config=config,
        metadata=metadata,
    )


def signal_power(samples: ComplexArray) -> float:
    """Kompleks sinyalin ortalama gücünü hesaplar."""

    samples = np.asarray(samples, dtype=np.complex128)

    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("samples tek boyutlu ve boş olmayan olmalıdır.")

    power = float(np.mean(np.abs(samples) ** 2))

    if not np.isfinite(power) or power < 0.0:
        raise ValueError("Geçerli bir sinyal gücü hesaplanamadı.")

    return power


def calculate_snr_db(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    """
    Reference sinyali doğru kabul ederek örnek tabanlı SNR hesaplar.

    noise = measured - reference
    """

    reference = np.asarray(reference, dtype=np.complex128)
    measured = np.asarray(measured, dtype=np.complex128)

    if reference.shape != measured.shape:
        raise ValueError("reference ve measured aynı boyutta olmalıdır.")

    reference_power = signal_power(reference)
    error_power = signal_power(measured - reference)

    if reference_power <= 0.0:
        raise ValueError("Referans sinyal gücü pozitif olmalıdır.")

    if error_power == 0.0:
        return float("inf")

    return float(10.0 * np.log10(reference_power / error_power))


def calculate_nmse(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    """Normalize edilmiş ortalama karesel hatayı hesaplar."""

    reference = np.asarray(reference, dtype=np.complex128)
    measured = np.asarray(measured, dtype=np.complex128)

    if reference.shape != measured.shape:
        raise ValueError("reference ve measured aynı boyutta olmalıdır.")

    denominator = float(np.sum(np.abs(reference) ** 2))

    if denominator <= 0.0:
        raise ValueError("Referans sinyal enerjisi pozitif olmalıdır.")

    numerator = float(np.sum(np.abs(measured - reference) ** 2))
    return numerator / denominator


def calculate_evm_rms_percent(
    reference_symbols: ComplexArray,
    measured_symbols: ComplexArray,
) -> float:
    """
    RMS EVM değerini yüzde olarak hesaplar.

    EVM(%) = 100 * sqrt(
        sum(|measured-reference|^2) / sum(|reference|^2)
    )
    """

    nmse = calculate_nmse(reference_symbols, measured_symbols)
    return float(100.0 * np.sqrt(nmse))


def calculate_papr_db(samples: ComplexArray) -> float:
    """Peak-to-average power ratio değerini dB cinsinden hesaplar."""

    average_power = signal_power(samples)

    if average_power <= 0.0:
        raise ValueError("Ortalama güç pozitif olmalıdır.")

    peak_power = float(np.max(np.abs(samples) ** 2))

    return float(10.0 * np.log10(peak_power / average_power))


def qpsk_hard_demodulate(symbols: ComplexArray) -> BitArray:
    """
    Adım 01 haritalamasıyla uyumlu QPSK hard-decision demodülasyonu.

    I < 0 -> ilk bit 1
    Q < 0 -> ikinci bit 1
    """

    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.ndim != 1:
        raise ValueError("symbols tek boyutlu olmalıdır.")

    bits = np.empty((len(symbols), 2), dtype=np.int8)
    bits[:, 0] = (symbols.real < 0.0).astype(np.int8)
    bits[:, 1] = (symbols.imag < 0.0).astype(np.int8)

    return bits


def calculate_ber(
    reference_bits: BitArray,
    estimated_bits: BitArray,
) -> tuple[float, int, int]:
    """Bit hata oranını, hata sayısını ve toplam bit sayısını döndürür."""

    reference_bits = np.asarray(reference_bits, dtype=np.int8)
    estimated_bits = np.asarray(estimated_bits, dtype=np.int8)

    if reference_bits.shape != estimated_bits.shape:
        raise ValueError(
            "reference_bits ve estimated_bits aynı boyutta olmalıdır."
        )

    if reference_bits.size == 0:
        raise ValueError("Bit dizileri boş olamaz.")

    if not np.all((reference_bits == 0) | (reference_bits == 1)):
        raise ValueError("reference_bits yalnızca 0 ve 1 içermelidir.")

    if not np.all((estimated_bits == 0) | (estimated_bits == 1)):
        raise ValueError("estimated_bits yalnızca 0 ve 1 içermelidir.")

    error_count = int(np.count_nonzero(reference_bits != estimated_bits))
    total_bits = int(reference_bits.size)
    ber = float(error_count / total_bits)

    return ber, error_count, total_bits


def estimate_best_complex_gain(
    reference: ComplexArray,
    measured: ComplexArray,
) -> complex:
    """
    measured ≈ gain * reference modeli için least-squares kompleks gain bulur.
    """

    reference = np.asarray(reference, dtype=np.complex128)
    measured = np.asarray(measured, dtype=np.complex128)

    if reference.shape != measured.shape:
        raise ValueError("reference ve measured aynı boyutta olmalıdır.")

    denominator = np.vdot(reference, reference)

    if np.abs(denominator) <= 1e-15:
        raise ValueError("Referans enerjisi gain tahmini için yetersiz.")

    gain = np.vdot(reference, measured) / denominator
    return complex(gain)


def equalize_single_complex_gain(
    reference: ComplexArray,
    measured: ComplexArray,
) -> tuple[ComplexArray, complex]:
    """Tek kompleks gain/phase katsayısını least-squares ile giderir."""

    gain = estimate_best_complex_gain(reference, measured)

    if abs(gain) <= 1e-12:
        raise RuntimeError("Tahmin edilen kompleks gain sıfıra çok yakın.")

    corrected = measured / gain
    return corrected.astype(np.complex128), gain


def read_rrc_group_delay_samples() -> int:
    """Adım 01 veya Adım 02 metadata içinden RRC grup gecikmesini okur."""

    if STEP01_METADATA.exists():
        document = load_json(STEP01_METADATA)

        if "rrc_group_delay_samples" in document:
            return int(document["rrc_group_delay_samples"])

    if STEP02_METADATA.exists():
        document = load_json(STEP02_METADATA)
        metadata = document.get("metadata", {})

        if "rrc_group_delay_samples" in metadata:
            return int(metadata["rrc_group_delay_samples"])

    raise FileNotFoundError(
        "RRC grup gecikmesi metadata içinde bulunamadı."
    )


def extract_symbol_samples(
    dataset: DatasetCase,
) -> tuple[ComplexArray, NDArray[np.int64], dict[str, int]]:
    """
    Bilinen sistem gecikmelerini kullanarak alınan sinyalden sembol örnekleri seçer.

    Bu timing recovery değildir. Ground-truth tabanlı ilk ölçüm noktasıdır.
    """

    rrc_group_delay = read_rrc_group_delay_samples()
    fractional_filter_group_delay = (
        len(dataset.fractional_delay_taps) - 1
    ) // 2
    dominant_channel_delay = int(
        np.argmax(np.abs(dataset.channel_taps))
    )

    first_symbol_index = (
        rrc_group_delay
        + fractional_filter_group_delay
        + dominant_channel_delay
    )

    number_of_symbols = len(dataset.symbols)

    indices = (
        first_symbol_index
        + np.arange(number_of_symbols, dtype=np.int64)
        * dataset.samples_per_symbol
    )

    valid_mask = indices < len(dataset.received_samples)
    indices = indices[valid_mask]

    if len(indices) < number_of_symbols // 2:
        raise RuntimeError("Yeterli sayıda sembol örneği çıkarılamadı.")

    samples = dataset.received_samples[indices]

    delays = {
        "rrc_group_delay_samples": rrc_group_delay,
        "fractional_filter_group_delay_samples": (
            fractional_filter_group_delay
        ),
        "dominant_channel_delay_samples": dominant_channel_delay,
        "first_symbol_index": first_symbol_index,
    }

    return (
        samples.astype(np.complex128),
        indices.astype(np.int64),
        delays,
    )


def oracle_carrier_correction(
    measured_symbol_samples: ComplexArray,
    output_indices: NDArray[np.int64],
    dataset: DatasetCase,
) -> ComplexArray:
    """
    Bilinen CFO ve faz değerlerini kullanarak oracle carrier düzeltmesi yapar.

    CFO, multipath ve timing filtresinden önce uygulandığı için çıkış indeksinden
    sonraki sabit FIR gecikmeleri çıkarılarak yaklaşık kaynak indeksi bulunur.
    """

    fractional_filter_group_delay = (
        len(dataset.fractional_delay_taps) - 1
    ) // 2
    dominant_channel_delay = int(
        np.argmax(np.abs(dataset.channel_taps))
    )

    source_indices = (
        output_indices
        - fractional_filter_group_delay
        - dominant_channel_delay
    ).astype(np.float64)

    phase_offset_radians = np.deg2rad(
        dataset.phase_offset_degrees
    )

    phase = (
        2.0
        * np.pi
        * dataset.cfo_hz
        * source_indices
        / dataset.sample_rate_hz
        + phase_offset_radians
    )

    corrected = measured_symbol_samples * np.exp(-1j * phase)
    return corrected.astype(np.complex128)


def run_awgn_self_test(
    reference_symbols: ComplexArray,
) -> dict[str, Any]:
    """Metrik fonksiyonlarını kontrollü QPSK + AWGN verisiyle doğrular."""

    rng = np.random.default_rng(SELF_TEST_SEED)
    reference_power = signal_power(reference_symbols)

    snr_linear = 10.0 ** (SELF_TEST_SNR_DB / 10.0)
    noise_power = reference_power / snr_linear
    component_std = np.sqrt(noise_power / 2.0)

    noise = component_std * (
        rng.standard_normal(len(reference_symbols))
        + 1j * rng.standard_normal(len(reference_symbols))
    )
    noise = noise.astype(np.complex128)

    measured = (reference_symbols + noise).astype(np.complex128)

    achieved_snr_db = calculate_snr_db(
        reference_symbols,
        measured,
    )
    evm_percent = calculate_evm_rms_percent(
        reference_symbols,
        measured,
    )
    estimated_bits = qpsk_hard_demodulate(measured)
    reference_bits = qpsk_hard_demodulate(reference_symbols)
    ber, errors, total_bits = calculate_ber(
        reference_bits,
        estimated_bits,
    )

    expected_evm_percent = float(
        100.0 * 10.0 ** (-SELF_TEST_SNR_DB / 20.0)
    )

    if abs(achieved_snr_db - SELF_TEST_SNR_DB) > 0.35:
        raise RuntimeError("AWGN öz testinde SNR toleransı aşıldı.")

    if abs(evm_percent - expected_evm_percent) > 1.0:
        raise RuntimeError("AWGN öz testinde EVM toleransı aşıldı.")

    return {
        "target_snr_db": SELF_TEST_SNR_DB,
        "achieved_snr_db": achieved_snr_db,
        "expected_evm_percent": expected_evm_percent,
        "measured_evm_percent": evm_percent,
        "ber": ber,
        "bit_errors": errors,
        "total_bits": total_bits,
        "status": "PASSED",
    }


def build_metrics(dataset: DatasetCase) -> dict[str, Any]:
    """Pipeline ve sembol seviyesinde bütün metrikleri hesaplar."""

    noise_stage_snr_db = calculate_snr_db(
        dataset.clipped_samples,
        dataset.received_samples,
    )
    noise_stage_nmse = calculate_nmse(
        dataset.clipped_samples,
        dataset.received_samples,
    )
    noise_stage_evm_percent = calculate_evm_rms_percent(
        dataset.clipped_samples,
        dataset.received_samples,
    )

    symbol_samples, symbol_indices, delays = extract_symbol_samples(
        dataset
    )

    symbol_count = min(len(symbol_samples), len(dataset.symbols))
    reference_symbols = dataset.symbols[:symbol_count]
    reference_bits = dataset.bits[:symbol_count]
    raw_symbols = symbol_samples[:symbol_count]

    raw_estimated_bits = qpsk_hard_demodulate(raw_symbols)
    raw_ber, raw_errors, raw_total_bits = calculate_ber(
        reference_bits,
        raw_estimated_bits,
    )
    raw_evm_percent = calculate_evm_rms_percent(
        reference_symbols,
        raw_symbols,
    )

    oracle_carrier_symbols = oracle_carrier_correction(
        measured_symbol_samples=raw_symbols,
        output_indices=symbol_indices[:symbol_count],
        dataset=dataset,
    )

    oracle_equalized_symbols, oracle_gain = (
        equalize_single_complex_gain(
            reference=reference_symbols,
            measured=oracle_carrier_symbols,
        )
    )

    oracle_estimated_bits = qpsk_hard_demodulate(
        oracle_equalized_symbols
    )
    oracle_ber, oracle_errors, oracle_total_bits = calculate_ber(
        reference_bits,
        oracle_estimated_bits,
    )
    oracle_evm_percent = calculate_evm_rms_percent(
        reference_symbols,
        oracle_equalized_symbols,
    )

    self_test = run_awgn_self_test(dataset.symbols)

    return {
        "noise_stage": {
            "description": (
                "Clipping sonrası gürültüsüz sinyal ile nihai alınan sinyal"
            ),
            "reference_power": signal_power(dataset.clipped_samples),
            "received_power": signal_power(dataset.received_samples),
            "noise_power": signal_power(dataset.noise_samples),
            "snr_db": noise_stage_snr_db,
            "target_snr_db": dataset.target_snr_db,
            "stored_achieved_snr_db": dataset.achieved_snr_db,
            "nmse": noise_stage_nmse,
            "evm_rms_percent": noise_stage_evm_percent,
            "papr_reference_db": calculate_papr_db(
                dataset.clipped_samples
            ),
            "papr_received_db": calculate_papr_db(
                dataset.received_samples
            ),
        },
        "pre_recovery_symbol_metrics": {
            "description": (
                "DSP kurtarma uygulanmadan, bilinen gecikmede alınan QPSK örnekleri"
            ),
            "symbol_count": symbol_count,
            "ber": raw_ber,
            "bit_errors": raw_errors,
            "total_bits": raw_total_bits,
            "evm_rms_percent": raw_evm_percent,
            "timing_information": delays,
        },
        "oracle_carrier_metrics": {
            "description": (
                "Bilinen CFO/faz ve tek kompleks gain düzeltmesi sonrası ölçüm"
            ),
            "ber": oracle_ber,
            "bit_errors": oracle_errors,
            "total_bits": oracle_total_bits,
            "evm_rms_percent": oracle_evm_percent,
            "estimated_complex_gain_real": float(oracle_gain.real),
            "estimated_complex_gain_imag": float(oracle_gain.imag),
            "estimated_complex_gain_magnitude": float(
                abs(oracle_gain)
            ),
            "estimated_complex_gain_phase_degrees": float(
                np.rad2deg(np.angle(oracle_gain))
            ),
        },
        "awgn_metric_self_test": self_test,
    }


def save_csv(metrics: dict[str, Any], output_path: Path) -> None:
    """Önemli metrikleri düz CSV tablosuna kaydeder."""

    rows = [
        (
            "noise_stage",
            "snr_db",
            metrics["noise_stage"]["snr_db"],
        ),
        (
            "noise_stage",
            "nmse",
            metrics["noise_stage"]["nmse"],
        ),
        (
            "noise_stage",
            "evm_rms_percent",
            metrics["noise_stage"]["evm_rms_percent"],
        ),
        (
            "pre_recovery",
            "ber",
            metrics["pre_recovery_symbol_metrics"]["ber"],
        ),
        (
            "pre_recovery",
            "evm_rms_percent",
            metrics["pre_recovery_symbol_metrics"][
                "evm_rms_percent"
            ],
        ),
        (
            "oracle_carrier",
            "ber",
            metrics["oracle_carrier_metrics"]["ber"],
        ),
        (
            "oracle_carrier",
            "evm_rms_percent",
            metrics["oracle_carrier_metrics"][
                "evm_rms_percent"
            ],
        ),
        (
            "awgn_self_test",
            "snr_db",
            metrics["awgn_metric_self_test"][
                "achieved_snr_db"
            ],
        ),
        (
            "awgn_self_test",
            "ber",
            metrics["awgn_metric_self_test"]["ber"],
        ),
        (
            "awgn_self_test",
            "evm_rms_percent",
            metrics["awgn_metric_self_test"][
                "measured_evm_percent"
            ],
        ),
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric_group", "metric_name", "value"])
        writer.writerows(rows)


def create_metrics_plot(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """BER, EVM ve SNR sonuçlarını görselleştirir."""

    figure, axes = plt.subplots(3, 1, figsize=(11, 13))

    evm_names = [
        "Noise stage",
        "Pre-recovery",
        "Oracle carrier",
        "AWGN self-test",
    ]
    evm_values = [
        metrics["noise_stage"]["evm_rms_percent"],
        metrics["pre_recovery_symbol_metrics"][
            "evm_rms_percent"
        ],
        metrics["oracle_carrier_metrics"]["evm_rms_percent"],
        metrics["awgn_metric_self_test"]["measured_evm_percent"],
    ]

    axes[0].bar(evm_names, evm_values)
    axes[0].set_title("RMS EVM Karşılaştırması")
    axes[0].set_ylabel("EVM (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y")

    ber_names = [
        "Pre-recovery",
        "Oracle carrier",
        "AWGN self-test",
    ]
    ber_values = [
        metrics["pre_recovery_symbol_metrics"]["ber"],
        metrics["oracle_carrier_metrics"]["ber"],
        metrics["awgn_metric_self_test"]["ber"],
    ]

    axes[1].bar(ber_names, ber_values)
    axes[1].set_title("QPSK Bit Hata Oranı")
    axes[1].set_ylabel("BER")
    axes[1].set_ylim(0.0, max(0.55, max(ber_values) * 1.1))
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y")

    snr_names = [
        "Target",
        "Stored",
        "Recalculated",
        "Self-test target",
        "Self-test measured",
    ]
    snr_values = [
        metrics["noise_stage"]["target_snr_db"],
        metrics["noise_stage"]["stored_achieved_snr_db"],
        metrics["noise_stage"]["snr_db"],
        metrics["awgn_metric_self_test"]["target_snr_db"],
        metrics["awgn_metric_self_test"]["achieved_snr_db"],
    ]

    axes[2].bar(snr_names, snr_values)
    axes[2].set_title("SNR Doğrulaması")
    axes[2].set_ylabel("SNR (dB)")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(True, axis="y")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def validate_metrics(
    dataset: DatasetCase,
    metrics: dict[str, Any],
) -> None:
    """Metrik motorunun sayısal tutarlılığını doğrular."""

    recalculated_snr = metrics["noise_stage"]["snr_db"]

    if abs(recalculated_snr - dataset.achieved_snr_db) > 1e-10:
        raise RuntimeError(
            "Yeniden hesaplanan SNR ile kayıtlı SNR uyuşmuyor."
        )

    noise_nmse = metrics["noise_stage"]["nmse"]
    noise_evm = metrics["noise_stage"]["evm_rms_percent"]

    expected_evm = 100.0 * np.sqrt(noise_nmse)

    if not np.isclose(noise_evm, expected_evm, atol=1e-12):
        raise RuntimeError("EVM ve NMSE bağıntısı doğrulanamadı.")

    for group_name in (
        "pre_recovery_symbol_metrics",
        "oracle_carrier_metrics",
    ):
        ber = float(metrics[group_name]["ber"])

        if not 0.0 <= ber <= 1.0:
            raise RuntimeError(f"{group_name} BER değeri geçersiz.")

    if metrics["awgn_metric_self_test"]["status"] != "PASSED":
        raise RuntimeError("AWGN metrik öz testi başarısız.")


def main() -> None:
    STEP07_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_case()
    metrics = build_metrics(dataset)

    validate_metrics(
        dataset=dataset,
        metrics=metrics,
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 7,
        "description": "Central SNR, NMSE, EVM, BER and PAPR engine",
        "input_step": 6,
        "metrics": metrics,
        "validations": {
            "stored_and_recalculated_snr_match": True,
            "evm_nmse_identity_verified": True,
            "ber_bounds_verified": True,
            "awgn_metric_self_test": "PASSED",
        },
    }

    json_path = STEP07_DIRECTORY / "metrics_report.json"
    csv_path = STEP07_DIRECTORY / "metrics_summary.csv"
    plot_path = STEP07_DIRECTORY / "metrics_overview.png"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    save_csv(metrics, csv_path)
    create_metrics_plot(metrics, plot_path)

    noise_metrics = metrics["noise_stage"]
    raw_metrics = metrics["pre_recovery_symbol_metrics"]
    oracle_metrics = metrics["oracle_carrier_metrics"]
    self_test = metrics["awgn_metric_self_test"]

    print()
    print("=" * 78)
    print("GENESIS-DSP — ADIM 07 BAŞARIYLA TAMAMLANDI")
    print("=" * 78)
    print(f"Yeniden hesaplanan SNR           : {noise_metrics['snr_db']:.6f} dB")
    print(f"Noise-stage NMSE                 : {noise_metrics['nmse']:.12e}")
    print(
        f"Noise-stage RMS EVM              : "
        f"{noise_metrics['evm_rms_percent']:.6f} %"
    )
    print(f"Kurtarma öncesi QPSK BER         : {raw_metrics['ber']:.9f}")
    print(
        f"Kurtarma öncesi bit hatası       : "
        f"{raw_metrics['bit_errors']} / {raw_metrics['total_bits']}"
    )
    print(
        f"Kurtarma öncesi RMS EVM          : "
        f"{raw_metrics['evm_rms_percent']:.6f} %"
    )
    print(f"Oracle carrier sonrası BER       : {oracle_metrics['ber']:.9f}")
    print(
        f"Oracle carrier sonrası RMS EVM   : "
        f"{oracle_metrics['evm_rms_percent']:.6f} %"
    )
    print(
        f"AWGN öz-test SNR                 : "
        f"{self_test['achieved_snr_db']:.6f} dB"
    )
    print(
        f"AWGN öz-test EVM                 : "
        f"{self_test['measured_evm_percent']:.6f} %"
    )
    print(f"AWGN öz-test durumu              : {self_test['status']}")
    print(f"JSON raporu                      : {json_path}")
    print(f"CSV özeti                        : {csv_path}")
    print(f"Grafik                           : {plot_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
