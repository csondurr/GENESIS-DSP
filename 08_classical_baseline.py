"""
GENESIS-DSP — Adım 08
Klasik DSP alıcı baseline'ı.

Bu program:
1. Adım 06 birleşik bozunum verisini yükler.
2. Örnek ortalamasıyla DC offset tahmini ve giderimi yapar.
3. QPSK dördüncü-kuvvet yöntemiyle CFO tahmini yapar.
4. CFO düzeltmesi ve RRC matched filtering uygular.
5. Training-aided timing başlangıcı arar.
6. Widely-linear FIR equalizer eğitir.
7. Eğitim, doğrulama ve test BER/EVM değerlerini raporlar.
8. Baseline veri paketi, JSON, CSV ve grafik üretir.

Çalıştırma:
    python 08_classical_baseline.py
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
STEP01_DIRECTORY = BASE_DIRECTORY / "outputs" / "step01"
STEP06_DIRECTORY = BASE_DIRECTORY / "outputs" / "step06"
STEP08_DIRECTORY = BASE_DIRECTORY / "outputs" / "step08"

INPUT_PACKAGE = STEP06_DIRECTORY / "dataset_case_0001.npz"
INPUT_CONFIG = STEP06_DIRECTORY / "pipeline_config.json"
RRC_TAPS_PATH = STEP01_DIRECTORY / "rrc_filter_taps.npy"

EQUALIZER_LENGTH = 11
TRAIN_SYMBOLS = 1200
VALIDATION_SYMBOLS = 600
TIMING_SEARCH_START = 0
TIMING_SEARCH_STOP = 128
RIDGE_REGULARIZATION = 1e-4

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.int8]


@dataclass(frozen=True)
class DatasetCase:
    received_samples: ComplexArray
    bits: BitArray
    symbols: ComplexArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    true_cfo_hz: float
    true_phase_offset_degrees: float
    target_snr_db: float
    rrc_taps: FloatArray


@dataclass(frozen=True)
class CandidateResult:
    timing_start: int
    validation_nmse: float
    validation_ber: float
    weights: ComplexArray
    predictions: ComplexArray
    reference_symbols: ComplexArray
    feature_half_length: int
    sampled_symbols: ComplexArray


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON dosyası bulunamadı: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset() -> DatasetCase:
    """Adım 06 paketini ve Adım 01 RRC filtresini yükler."""

    if not INPUT_PACKAGE.exists():
        raise FileNotFoundError(
            f"Adım 06 veri paketi bulunamadı: {INPUT_PACKAGE}\n"
            "Önce şu komutu çalıştır:\n"
            "python 06_impairment_pipeline.py"
        )

    if not INPUT_CONFIG.exists():
        raise FileNotFoundError(
            f"Pipeline config bulunamadı: {INPUT_CONFIG}"
        )

    if not RRC_TAPS_PATH.exists():
        raise FileNotFoundError(
            f"RRC filtresi bulunamadı: {RRC_TAPS_PATH}\n"
            "Önce şu komutu çalıştır:\n"
            "python 01_qpsk_generator.py"
        )

    config_document = load_json(INPUT_CONFIG)
    config = config_document["config"]

    with np.load(INPUT_PACKAGE, allow_pickle=False) as package:
        received_samples = package["received_samples"].astype(
            np.complex128
        )
        bits = package["bits"].astype(np.int8)
        symbols = package["symbols"].astype(np.complex128)
        sample_rate_hz = float(package["sample_rate_hz"])
        symbol_rate_hz = float(package["symbol_rate_hz"])
        samples_per_symbol = int(package["samples_per_symbol"])

    rrc_taps = np.load(
        RRC_TAPS_PATH,
        allow_pickle=False,
    ).astype(np.float64)

    if received_samples.ndim != 1 or len(received_samples) == 0:
        raise ValueError(
            "received_samples tek boyutlu ve boş olmayan olmalıdır."
        )

    if not np.all(np.isfinite(received_samples.real)):
        raise ValueError(
            "received_samples gerçek kısmında NaN veya Inf bulundu."
        )

    if not np.all(np.isfinite(received_samples.imag)):
        raise ValueError(
            "received_samples sanal kısmında NaN veya Inf bulundu."
        )

    if bits.ndim != 2 or bits.shape[1] != 2:
        raise ValueError(
            "bits dizisi (sembol_sayısı, 2) biçiminde olmalıdır."
        )

    if symbols.ndim != 1 or len(symbols) != len(bits):
        raise ValueError(
            "symbols boyutu bit çifti sayısıyla uyumlu değil."
        )

    if EQUALIZER_LENGTH < 3 or EQUALIZER_LENGTH % 2 == 0:
        raise ValueError(
            "EQUALIZER_LENGTH en az 3 ve tek sayı olmalıdır."
        )

    return DatasetCase(
        received_samples=received_samples,
        bits=bits,
        symbols=symbols,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=samples_per_symbol,
        true_cfo_hz=float(config["cfo_hz"]),
        true_phase_offset_degrees=float(
            config["phase_offset_degrees"]
        ),
        target_snr_db=float(config["target_snr_db"]),
        rrc_taps=rrc_taps,
    )


def signal_power(samples: ComplexArray) -> float:
    power = float(np.mean(np.abs(samples) ** 2))

    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("Sinyal gücü pozitif ve sonlu olmalıdır.")

    return power


def estimate_and_remove_dc(
    samples: ComplexArray,
) -> tuple[ComplexArray, complex]:
    """Örnek ortalamasını kompleks DC offset tahmini kabul eder."""

    estimated_dc = complex(np.mean(samples))
    corrected = (samples - estimated_dc).astype(np.complex128)

    return corrected, estimated_dc


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value pozitif olmalıdır.")

    return 1 << (value - 1).bit_length()


def parabolic_peak_offset(
    left_value: float,
    center_value: float,
    right_value: float,
) -> float:
    """
    FFT bin tepesinin alt-bin konumunu parabolik enterpolasyonla tahmin eder.
    """

    denominator = left_value - 2.0 * center_value + right_value

    if abs(denominator) <= 1e-30:
        return 0.0

    offset = 0.5 * (left_value - right_value) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def estimate_qpsk_cfo_fourth_power(
    samples: ComplexArray,
    sample_rate_hz: float,
) -> tuple[float, FloatArray, FloatArray]:
    """
    QPSK dördüncü-kuvvet CFO tahmini.

    QPSK veri fazı dördüncü kuvvette büyük ölçüde kaldırılır.
    Oluşan spektral tepe yaklaşık 4*CFO frekansındadır.
    """

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz pozitif olmalıdır.")

    fourth_power = samples ** 4
    window = np.hanning(len(fourth_power))

    fft_size = next_power_of_two(
        max(65_536, 4 * len(fourth_power))
    )

    spectrum = np.fft.fftshift(
        np.fft.fft(
            fourth_power * window,
            n=fft_size,
        )
    )

    magnitude_squared = np.abs(spectrum) ** 2
    frequencies_hz = np.fft.fftshift(
        np.fft.fftfreq(
            fft_size,
            d=1.0 / sample_rate_hz,
        )
    )

    peak_index = int(np.argmax(magnitude_squared))
    refined_peak_index = float(peak_index)

    if 0 < peak_index < len(magnitude_squared) - 1:
        offset = parabolic_peak_offset(
            float(magnitude_squared[peak_index - 1]),
            float(magnitude_squared[peak_index]),
            float(magnitude_squared[peak_index + 1]),
        )
        refined_peak_index += offset

    frequency_step_hz = sample_rate_hz / fft_size
    shifted_zero_index = fft_size / 2.0

    fourth_power_frequency_hz = (
        refined_peak_index - shifted_zero_index
    ) * frequency_step_hz

    estimated_cfo_hz = fourth_power_frequency_hz / 4.0

    normalized_spectrum = magnitude_squared / float(
        np.max(magnitude_squared)
    )
    spectrum_db = 10.0 * np.log10(
        np.maximum(normalized_spectrum, 1e-15)
    )

    return (
        float(estimated_cfo_hz),
        frequencies_hz.astype(np.float64),
        spectrum_db.astype(np.float64),
    )


def correct_cfo(
    samples: ComplexArray,
    sample_rate_hz: float,
    estimated_cfo_hz: float,
) -> ComplexArray:
    """Tahmin edilen CFO değerini kompleks rotasyonla giderir."""

    indices = np.arange(len(samples), dtype=np.float64)
    correction = np.exp(
        -1j
        * 2.0
        * np.pi
        * estimated_cfo_hz
        * indices
        / sample_rate_hz
    )

    return (samples * correction).astype(np.complex128)


def matched_filter(
    samples: ComplexArray,
    rrc_taps: FloatArray,
) -> ComplexArray:
    """RRC matched filter uygular."""

    return np.convolve(
        samples,
        rrc_taps,
        mode="full",
    ).astype(np.complex128)


def build_widely_linear_features(
    sampled_symbols: ComplexArray,
    equalizer_length: int,
) -> tuple[ComplexArray, int]:
    """
    FIR, conjugate-FIR ve sabit terimden oluşan widely-linear özellik matrisi.

    Her satır:
        [x[k-Lh], ..., x[k+Lh],
         conj(x[k-Lh]), ..., conj(x[k+Lh]),
         1]
    """

    if equalizer_length % 2 == 0:
        raise ValueError("equalizer_length tek sayı olmalıdır.")

    if len(sampled_symbols) < equalizer_length:
        raise ValueError(
            "Equalizer için yeterli sayıda sembol örneği yok."
        )

    row_count = len(sampled_symbols) - equalizer_length + 1

    windows = np.stack(
        [
            sampled_symbols[
                tap_index:tap_index + row_count
            ]
            for tap_index in range(equalizer_length)
        ],
        axis=1,
    )

    constant_column = np.ones(
        (row_count, 1),
        dtype=np.complex128,
    )

    features = np.concatenate(
        [
            windows,
            np.conj(windows),
            constant_column,
        ],
        axis=1,
    )

    half_length = equalizer_length // 2

    return (
        features.astype(np.complex128),
        half_length,
    )


def fit_complex_ridge(
    features: ComplexArray,
    targets: ComplexArray,
    regularization: float,
) -> ComplexArray:
    """Kompleks ridge regression ile equalizer katsayılarını bulur."""

    if regularization < 0.0:
        raise ValueError("regularization negatif olamaz.")

    gram = features.conj().T @ features
    right_side = features.conj().T @ targets

    regularizer = regularization * np.eye(
        gram.shape[0],
        dtype=np.complex128,
    )

    weights = np.linalg.solve(
        gram + regularizer,
        right_side,
    )

    return weights.astype(np.complex128)


def qpsk_hard_demodulate(
    symbols: ComplexArray,
) -> BitArray:
    """Adım 01 QPSK eşlemesiyle uyumlu hard decision demodülasyonu."""

    bits = np.empty(
        (len(symbols), 2),
        dtype=np.int8,
    )

    bits[:, 0] = (symbols.real < 0.0).astype(np.int8)
    bits[:, 1] = (symbols.imag < 0.0).astype(np.int8)

    return bits


def calculate_ber(
    reference_bits: BitArray,
    estimated_bits: BitArray,
) -> tuple[float, int, int]:
    if reference_bits.shape != estimated_bits.shape:
        raise ValueError(
            "reference_bits ve estimated_bits aynı boyutta olmalıdır."
        )

    error_count = int(
        np.count_nonzero(
            reference_bits != estimated_bits
        )
    )
    total_bits = int(reference_bits.size)
    ber = float(error_count / total_bits)

    return ber, error_count, total_bits


def calculate_nmse(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    if reference.shape != measured.shape:
        raise ValueError(
            "reference ve measured aynı boyutta olmalıdır."
        )

    denominator = float(
        np.sum(np.abs(reference) ** 2)
    )

    if denominator <= 0.0:
        raise ValueError("Referans enerjisi pozitif olmalıdır.")

    numerator = float(
        np.sum(np.abs(measured - reference) ** 2)
    )

    return numerator / denominator


def calculate_evm_percent(
    reference: ComplexArray,
    measured: ComplexArray,
) -> float:
    return float(
        100.0
        * np.sqrt(
            calculate_nmse(reference, measured)
        )
    )


def evaluate_candidate(
    matched_samples: ComplexArray,
    dataset: DatasetCase,
    timing_start: int,
) -> CandidateResult | None:
    """Bir timing başlangıç adayı için equalizer eğitip validation skoru üretir."""

    indices = (
        timing_start
        + np.arange(
            len(dataset.symbols),
            dtype=np.int64,
        )
        * dataset.samples_per_symbol
    )

    indices = indices[
        indices < len(matched_samples)
    ]

    sampled_symbols = matched_samples[indices]

    if len(sampled_symbols) <= (
        EQUALIZER_LENGTH
        + TRAIN_SYMBOLS
        + VALIDATION_SYMBOLS
    ):
        return None

    features, half_length = build_widely_linear_features(
        sampled_symbols=sampled_symbols,
        equalizer_length=EQUALIZER_LENGTH,
    )

    reference_symbols = dataset.symbols[
        half_length:half_length + len(features)
    ]

    if len(reference_symbols) != len(features):
        return None

    train_end = TRAIN_SYMBOLS
    validation_end = TRAIN_SYMBOLS + VALIDATION_SYMBOLS

    train_features = features[:train_end]
    train_targets = reference_symbols[:train_end]

    weights = fit_complex_ridge(
        features=train_features,
        targets=train_targets,
        regularization=RIDGE_REGULARIZATION,
    )

    predictions = (features @ weights).astype(
        np.complex128
    )

    validation_reference = reference_symbols[
        train_end:validation_end
    ]
    validation_predictions = predictions[
        train_end:validation_end
    ]

    validation_bits = dataset.bits[
        half_length + train_end:
        half_length + validation_end
    ]

    validation_estimated_bits = qpsk_hard_demodulate(
        validation_predictions
    )

    validation_ber, _, _ = calculate_ber(
        validation_bits,
        validation_estimated_bits,
    )

    validation_nmse = calculate_nmse(
        validation_reference,
        validation_predictions,
    )

    return CandidateResult(
        timing_start=timing_start,
        validation_nmse=validation_nmse,
        validation_ber=validation_ber,
        weights=weights,
        predictions=predictions,
        reference_symbols=reference_symbols,
        feature_half_length=half_length,
        sampled_symbols=sampled_symbols,
    )


def search_best_timing_and_equalizer(
    matched_samples: ComplexArray,
    dataset: DatasetCase,
) -> tuple[CandidateResult, list[dict[str, float]]]:
    """Timing başlangıç adaylarını tarar ve en düşük validation NMSE'yi seçer."""

    candidates: list[CandidateResult] = []
    search_log: list[dict[str, float]] = []

    for timing_start in range(
        TIMING_SEARCH_START,
        TIMING_SEARCH_STOP + 1,
    ):
        candidate = evaluate_candidate(
            matched_samples=matched_samples,
            dataset=dataset,
            timing_start=timing_start,
        )

        if candidate is None:
            continue

        candidates.append(candidate)
        search_log.append(
            {
                "timing_start": float(
                    candidate.timing_start
                ),
                "validation_nmse": float(
                    candidate.validation_nmse
                ),
                "validation_ber": float(
                    candidate.validation_ber
                ),
            }
        )

    if not candidates:
        raise RuntimeError(
            "Geçerli timing/equalizer adayı oluşturulamadı."
        )

    best_candidate = min(
        candidates,
        key=lambda item: (
            item.validation_nmse,
            item.validation_ber,
            item.timing_start,
        ),
    )

    return best_candidate, search_log


def evaluate_split(
    name: str,
    predictions: ComplexArray,
    reference_symbols: ComplexArray,
    dataset: DatasetCase,
    half_length: int,
    start: int,
    stop: int,
) -> dict[str, Any]:
    """Belirli sembol aralığı için BER, EVM ve NMSE hesaplar."""

    actual_stop = min(
        stop,
        len(predictions),
        len(reference_symbols),
    )

    if actual_stop <= start:
        raise RuntimeError(
            f"{name} bölümü için geçerli örnek yok."
        )

    split_predictions = predictions[start:actual_stop]
    split_reference = reference_symbols[start:actual_stop]
    split_reference_bits = dataset.bits[
        half_length + start:
        half_length + actual_stop
    ]

    split_estimated_bits = qpsk_hard_demodulate(
        split_predictions
    )

    ber, bit_errors, total_bits = calculate_ber(
        split_reference_bits,
        split_estimated_bits,
    )

    nmse = calculate_nmse(
        split_reference,
        split_predictions,
    )

    evm_percent = calculate_evm_percent(
        split_reference,
        split_predictions,
    )

    return {
        "name": name,
        "start_symbol": start,
        "stop_symbol_exclusive": actual_stop,
        "symbol_count": int(actual_stop - start),
        "ber": ber,
        "bit_errors": bit_errors,
        "total_bits": total_bits,
        "nmse": nmse,
        "evm_rms_percent": evm_percent,
    }


def create_plots(
    dataset: DatasetCase,
    estimated_dc: complex,
    cfo_frequencies_hz: FloatArray,
    cfo_spectrum_db: FloatArray,
    estimated_cfo_hz: float,
    best_candidate: CandidateResult,
    search_log: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Baseline sonuçlarını dört grafikle görselleştirir."""

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(12, 18),
    )

    cfo_display_mask = np.abs(
        cfo_frequencies_hz
    ) <= 100_000.0

    axes[0].plot(
        cfo_frequencies_hz[cfo_display_mask] / 1_000.0,
        cfo_spectrum_db[cfo_display_mask],
    )
    axes[0].axvline(
        4.0 * estimated_cfo_hz / 1_000.0,
        linestyle="--",
        label="Tahmin edilen 4×CFO",
    )
    axes[0].set_title("QPSK Dördüncü-Kuvvet CFO Spektrumu")
    axes[0].set_xlabel("Frekans (kHz)")
    axes[0].set_ylabel("Normalize güç (dB)")
    axes[0].set_ylim(-100.0, 5.0)
    axes[0].grid(True)
    axes[0].legend()

    timing_values = [
        int(item["timing_start"])
        for item in search_log
    ]
    validation_nmse_values = [
        float(item["validation_nmse"])
        for item in search_log
    ]

    axes[1].plot(
        timing_values,
        validation_nmse_values,
    )
    axes[1].axvline(
        best_candidate.timing_start,
        linestyle="--",
        label="Seçilen timing",
    )
    axes[1].set_title("Timing Başlangıcı Arama Skoru")
    axes[1].set_xlabel("İlk örnek indeksi")
    axes[1].set_ylabel("Validation NMSE")
    axes[1].grid(True)
    axes[1].legend()

    plot_count = min(
        2500,
        len(best_candidate.sampled_symbols),
        len(best_candidate.predictions),
    )

    axes[2].scatter(
        best_candidate.sampled_symbols[:plot_count].real,
        best_candidate.sampled_symbols[:plot_count].imag,
        s=6,
        alpha=0.25,
        label="Equalizer öncesi",
    )
    axes[2].scatter(
        best_candidate.predictions[:plot_count].real,
        best_candidate.predictions[:plot_count].imag,
        s=7,
        alpha=0.35,
        label="Equalizer sonrası",
    )
    axes[2].set_title("Klasik Baseline Konstelasyonu")
    axes[2].set_xlabel("I")
    axes[2].set_ylabel("Q")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(True)
    axes[2].legend()

    error_magnitude = np.abs(
        best_candidate.predictions
        - best_candidate.reference_symbols
    )

    axes[3].plot(error_magnitude)
    axes[3].axvline(
        TRAIN_SYMBOLS,
        linestyle="--",
        label="Train sonu",
    )
    axes[3].axvline(
        TRAIN_SYMBOLS + VALIDATION_SYMBOLS,
        linestyle="--",
        label="Validation sonu",
    )
    axes[3].set_title(
        "Sembol Hata Büyüklüğü — Train/Validation/Test"
    )
    axes[3].set_xlabel("Equalizer çıkış sembol indeksi")
    axes[3].set_ylabel("|tahmin - referans|")
    axes[3].grid(True)
    axes[3].legend()

    figure.suptitle(
        (
            "GENESIS-DSP Classical Baseline\n"
            f"Tahmini DC={estimated_dc.real:+.5f}"
            f"{estimated_dc.imag:+.5f}j, "
            f"Tahmini CFO={estimated_cfo_hz:.3f} Hz"
        ),
        fontsize=13,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_csv(
    splits: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Train, validation ve test metriklerini CSV'ye kaydeder."""

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "start_symbol",
                "stop_symbol_exclusive",
                "symbol_count",
                "ber",
                "bit_errors",
                "total_bits",
                "nmse",
                "evm_rms_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(splits)


def main() -> None:
    STEP08_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset()

    dc_corrected_samples, estimated_dc = (
        estimate_and_remove_dc(
            dataset.received_samples
        )
    )

    (
        estimated_cfo_hz,
        cfo_frequencies_hz,
        cfo_spectrum_db,
    ) = estimate_qpsk_cfo_fourth_power(
        samples=dc_corrected_samples,
        sample_rate_hz=dataset.sample_rate_hz,
    )

    cfo_corrected_samples = correct_cfo(
        samples=dc_corrected_samples,
        sample_rate_hz=dataset.sample_rate_hz,
        estimated_cfo_hz=estimated_cfo_hz,
    )

    matched_samples = matched_filter(
        samples=cfo_corrected_samples,
        rrc_taps=dataset.rrc_taps,
    )

    best_candidate, search_log = (
        search_best_timing_and_equalizer(
            matched_samples=matched_samples,
            dataset=dataset,
        )
    )

    train_metrics = evaluate_split(
        name="train",
        predictions=best_candidate.predictions,
        reference_symbols=best_candidate.reference_symbols,
        dataset=dataset,
        half_length=best_candidate.feature_half_length,
        start=0,
        stop=TRAIN_SYMBOLS,
    )

    validation_metrics = evaluate_split(
        name="validation",
        predictions=best_candidate.predictions,
        reference_symbols=best_candidate.reference_symbols,
        dataset=dataset,
        half_length=best_candidate.feature_half_length,
        start=TRAIN_SYMBOLS,
        stop=TRAIN_SYMBOLS + VALIDATION_SYMBOLS,
    )

    test_metrics = evaluate_split(
        name="test",
        predictions=best_candidate.predictions,
        reference_symbols=best_candidate.reference_symbols,
        dataset=dataset,
        half_length=best_candidate.feature_half_length,
        start=TRAIN_SYMBOLS + VALIDATION_SYMBOLS,
        stop=len(best_candidate.predictions),
    )

    cfo_error_hz = estimated_cfo_hz - dataset.true_cfo_hz

    if abs(cfo_error_hz) > 50.0:
        raise RuntimeError(
            "Dördüncü-kuvvet CFO tahmini beklenen tolerans dışında."
        )

    if validation_metrics["ber"] > 0.10:
        raise RuntimeError(
            "Validation BER baseline başarı eşiğinin üzerinde."
        )

    if test_metrics["ber"] > 0.10:
        raise RuntimeError(
            "Test BER baseline başarı eşiğinin üzerinde."
        )

    if not np.all(
        np.isfinite(
            best_candidate.predictions.real
        )
    ):
        raise RuntimeError(
            "Equalizer çıkışının gerçek kısmında NaN veya Inf var."
        )

    if not np.all(
        np.isfinite(
            best_candidate.predictions.imag
        )
    ):
        raise RuntimeError(
            "Equalizer çıkışının sanal kısmında NaN veya Inf var."
        )

    np.savez_compressed(
        STEP08_DIRECTORY / "classical_baseline_record.npz",
        received_samples=dataset.received_samples,
        dc_corrected_samples=dc_corrected_samples,
        cfo_corrected_samples=cfo_corrected_samples,
        matched_filter_samples=matched_samples,
        sampled_symbols=best_candidate.sampled_symbols,
        equalized_symbols=best_candidate.predictions,
        reference_symbols=best_candidate.reference_symbols,
        equalizer_weights=best_candidate.weights,
        rrc_taps=dataset.rrc_taps,
        estimated_dc=np.complex128(estimated_dc),
        estimated_cfo_hz=np.float64(
            estimated_cfo_hz
        ),
        true_cfo_hz=np.float64(
            dataset.true_cfo_hz
        ),
        selected_timing_start=np.int64(
            best_candidate.timing_start
        ),
        equalizer_length=np.int64(
            EQUALIZER_LENGTH
        ),
        train_symbols=np.int64(
            TRAIN_SYMBOLS
        ),
        validation_symbols=np.int64(
            VALIDATION_SYMBOLS
        ),
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 8,
        "description": (
            "Classical QPSK receiver baseline with DC removal, "
            "fourth-power CFO estimation, matched filtering, "
            "training-aided timing search and widely-linear FIR equalization"
        ),
        "input_step": 6,
        "receiver": {
            "estimated_dc_real": float(
                estimated_dc.real
            ),
            "estimated_dc_imag": float(
                estimated_dc.imag
            ),
            "true_cfo_hz": dataset.true_cfo_hz,
            "estimated_cfo_hz": estimated_cfo_hz,
            "cfo_error_hz": cfo_error_hz,
            "selected_timing_start": (
                best_candidate.timing_start
            ),
            "equalizer_length": EQUALIZER_LENGTH,
            "equalizer_feature_count": int(
                len(best_candidate.weights)
            ),
            "ridge_regularization": (
                RIDGE_REGULARIZATION
            ),
            "training_symbols": TRAIN_SYMBOLS,
            "validation_symbols": (
                VALIDATION_SYMBOLS
            ),
            "test_symbols": test_metrics[
                "symbol_count"
            ],
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "timing_search": search_log,
        "validations": {
            "cfo_error_below_50_hz": True,
            "validation_ber_below_0_10": True,
            "test_ber_below_0_10": True,
            "finite_equalizer_output": True,
        },
    }

    json_path = STEP08_DIRECTORY / "baseline_report.json"
    csv_path = STEP08_DIRECTORY / "baseline_metrics.csv"
    plot_path = STEP08_DIRECTORY / "baseline_overview.png"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    save_csv(
        splits=[
            train_metrics,
            validation_metrics,
            test_metrics,
        ],
        output_path=csv_path,
    )

    create_plots(
        dataset=dataset,
        estimated_dc=estimated_dc,
        cfo_frequencies_hz=cfo_frequencies_hz,
        cfo_spectrum_db=cfo_spectrum_db,
        estimated_cfo_hz=estimated_cfo_hz,
        best_candidate=best_candidate,
        search_log=search_log,
        output_path=plot_path,
    )

    print()
    print("=" * 82)
    print("GENESIS-DSP — ADIM 08 BAŞARIYLA TAMAMLANDI")
    print("=" * 82)
    print(
        f"Tahmini DC offset                 : "
        f"{estimated_dc.real:+.8f} "
        f"{estimated_dc.imag:+.8f}j"
    )
    print(
        f"Gerçek CFO                        : "
        f"{dataset.true_cfo_hz:,.6f} Hz"
    )
    print(
        f"Tahmini CFO                       : "
        f"{estimated_cfo_hz:,.6f} Hz"
    )
    print(
        f"CFO tahmin hatası                 : "
        f"{cfo_error_hz:+.6f} Hz"
    )
    print(
        f"Seçilen timing başlangıcı         : "
        f"{best_candidate.timing_start}"
    )
    print(
        f"Equalizer uzunluğu                : "
        f"{EQUALIZER_LENGTH} sembol tap"
    )
    print(
        f"Train BER                         : "
        f"{train_metrics['ber']:.9f}"
    )
    print(
        f"Validation BER                    : "
        f"{validation_metrics['ber']:.9f}"
    )
    print(
        f"Test BER                          : "
        f"{test_metrics['ber']:.9f}"
    )
    print(
        f"Test bit hatası                   : "
        f"{test_metrics['bit_errors']} / "
        f"{test_metrics['total_bits']}"
    )
    print(
        f"Test RMS EVM                      : "
        f"{test_metrics['evm_rms_percent']:.6f} %"
    )
    print(f"JSON raporu                       : {json_path}")
    print(f"CSV özeti                         : {csv_path}")
    print(f"Grafik                            : {plot_path}")
    print("=" * 82)


if __name__ == "__main__":
    main()
