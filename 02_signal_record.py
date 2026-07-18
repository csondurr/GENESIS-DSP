

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
STEP01_DIRECTORY = BASE_DIRECTORY / "outputs" / "step01"
STEP02_DIRECTORY = BASE_DIRECTORY / "outputs" / "step02"

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.int8]


def _sha256_array(array: NDArray[Any]) -> str:
    """Bir NumPy dizisinin byte içeriği için SHA-256 özeti üretir."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    """NumPy türlerini JSON ile uyumlu Python türlerine dönüştürür."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


@dataclass
class SignalRecord:
    """
    GENESIS-DSP içindeki bütün sinyal bloklarının kullanacağı ortak veri modeli.

    samples:
        Kompleks taban bant örnekleri.
    sample_rate_hz:
        Örnekleme frekansı.
    symbol_rate_hz:
        Sembol frekansı.
    samples_per_symbol:
        Sembol başına örnek sayısı.
    bits:
        Ground-truth bit çiftleri.
    symbols:
        Ground-truth kompleks semboller.
    metadata:
        Deney ve üretim bilgileri.
    """

    samples: ComplexArray
    sample_rate_hz: float
    symbol_rate_hz: float
    samples_per_symbol: int
    bits: BitArray | None = None
    symbols: ComplexArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Kayıt içindeki bütün temel fiziksel ve sayısal koşulları doğrular."""

        self.samples = np.asarray(self.samples, dtype=np.complex128)

        if self.samples.ndim != 1:
            raise ValueError("samples tek boyutlu olmalıdır.")

        if len(self.samples) == 0:
            raise ValueError("samples boş olamaz.")

        if not np.all(np.isfinite(self.samples.real)):
            raise ValueError("samples gerçek bileşeninde NaN veya Inf bulundu.")

        if not np.all(np.isfinite(self.samples.imag)):
            raise ValueError("samples sanal bileşeninde NaN veya Inf bulundu.")

        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz pozitif ve sonlu olmalıdır.")

        if not np.isfinite(self.symbol_rate_hz) or self.symbol_rate_hz <= 0.0:
            raise ValueError("symbol_rate_hz pozitif ve sonlu olmalıdır.")

        if self.samples_per_symbol <= 0:
            raise ValueError("samples_per_symbol pozitif olmalıdır.")

        expected_ratio = self.sample_rate_hz / self.symbol_rate_hz

        if not np.isclose(
            expected_ratio,
            float(self.samples_per_symbol),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "sample_rate_hz / symbol_rate_hz oranı "
                "samples_per_symbol ile uyumlu değil."
            )

        average_power = self.average_power

        if not np.isfinite(average_power) or average_power <= 0.0:
            raise ValueError("Sinyalin ortalama gücü pozitif ve sonlu olmalıdır.")

        if self.bits is not None:
            self.bits = np.asarray(self.bits, dtype=np.int8)

            if self.bits.ndim != 2 or self.bits.shape[1] != 2:
                raise ValueError(
                    "QPSK bits dizisi (sembol_sayısı, 2) boyutunda olmalıdır."
                )

            if not np.all((self.bits == 0) | (self.bits == 1)):
                raise ValueError("bits yalnızca 0 ve 1 içermelidir.")

        if self.symbols is not None:
            self.symbols = np.asarray(self.symbols, dtype=np.complex128)

            if self.symbols.ndim != 1:
                raise ValueError("symbols tek boyutlu olmalıdır.")

            if len(self.symbols) == 0:
                raise ValueError("symbols boş olamaz.")

            if not np.all(np.isfinite(self.symbols.real)):
                raise ValueError("symbols gerçek bileşeninde NaN veya Inf bulundu.")

            if not np.all(np.isfinite(self.symbols.imag)):
                raise ValueError("symbols sanal bileşeninde NaN veya Inf bulundu.")

        if self.bits is not None and self.symbols is not None:
            if len(self.bits) != len(self.symbols):
                raise ValueError(
                    "Bit çifti sayısı ile QPSK sembol sayısı eşit olmalıdır."
                )

    @property
    def number_of_samples(self) -> int:
        return int(len(self.samples))

    @property
    def duration_seconds(self) -> float:
        return float(self.number_of_samples / self.sample_rate_hz)

    @property
    def average_power(self) -> float:
        return float(np.mean(np.abs(self.samples) ** 2))

    @property
    def peak_magnitude(self) -> float:
        return float(np.max(np.abs(self.samples)))

    @property
    def rms_magnitude(self) -> float:
        return float(np.sqrt(self.average_power))

    @property
    def crest_factor_db(self) -> float:
        if self.rms_magnitude <= 0.0:
            return float("inf")

        return float(
            20.0 * np.log10(self.peak_magnitude / self.rms_magnitude)
        )

    def time_axis_seconds(self) -> FloatArray:
        return (
            np.arange(self.number_of_samples, dtype=np.float64)
            / self.sample_rate_hz
        )

    def summary(self) -> dict[str, Any]:
        """Kayıt için insan ve makine tarafından okunabilir özet üretir."""
        return {
            "number_of_samples": self.number_of_samples,
            "sample_rate_hz": self.sample_rate_hz,
            "symbol_rate_hz": self.symbol_rate_hz,
            "samples_per_symbol": self.samples_per_symbol,
            "duration_seconds": self.duration_seconds,
            "average_power": self.average_power,
            "rms_magnitude": self.rms_magnitude,
            "peak_magnitude": self.peak_magnitude,
            "crest_factor_db": self.crest_factor_db,
            "sample_dtype": str(self.samples.dtype),
            "number_of_symbols": (
                int(len(self.symbols)) if self.symbols is not None else None
            ),
            "number_of_bit_pairs": (
                int(len(self.bits)) if self.bits is not None else None
            ),
            "samples_sha256": _sha256_array(self.samples),
            "symbols_sha256": (
                _sha256_array(self.symbols)
                if self.symbols is not None
                else None
            ),
            "bits_sha256": (
                _sha256_array(self.bits)
                if self.bits is not None
                else None
            ),
        }

    def save(self, output_directory: Path) -> tuple[Path, Path]:
        """
        SignalRecord verisini NPZ ve JSON biçiminde kaydeder.

        Dönen değer:
            (npz_path, metadata_path)
        """
        self.validate()
        output_directory.mkdir(parents=True, exist_ok=True)

        npz_path = output_directory / "signal_record.npz"
        metadata_path = output_directory / "signal_record_metadata.json"

        bits_to_save = (
            self.bits
            if self.bits is not None
            else np.empty((0, 2), dtype=np.int8)
        )

        symbols_to_save = (
            self.symbols
            if self.symbols is not None
            else np.empty(0, dtype=np.complex128)
        )

        np.savez_compressed(
            npz_path,
            samples=self.samples,
            bits=bits_to_save,
            symbols=symbols_to_save,
            sample_rate_hz=np.float64(self.sample_rate_hz),
            symbol_rate_hz=np.float64(self.symbol_rate_hz),
            samples_per_symbol=np.int64(self.samples_per_symbol),
        )

        document = {
            "schema_name": "GENESIS-DSP SignalRecord",
            "schema_version": "1.0.0",
            "summary": self.summary(),
            "metadata": _json_safe(self.metadata),
        }

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                document,
                file,
                indent=4,
                ensure_ascii=False,
            )

        return npz_path, metadata_path

    @classmethod
    def load(
        cls,
        npz_path: Path,
        metadata_path: Path,
    ) -> "SignalRecord":
        """Kaydedilmiş SignalRecord paketini yükler ve doğrular."""

        if not npz_path.exists():
            raise FileNotFoundError(f"Veri paketi bulunamadı: {npz_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata bulunamadı: {metadata_path}")

        with np.load(npz_path, allow_pickle=False) as package:
            samples = package["samples"].astype(np.complex128)
            bits_raw = package["bits"].astype(np.int8)
            symbols_raw = package["symbols"].astype(np.complex128)

            sample_rate_hz = float(package["sample_rate_hz"])
            symbol_rate_hz = float(package["symbol_rate_hz"])
            samples_per_symbol = int(package["samples_per_symbol"])

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata_document = json.load(file)

        bits = bits_raw if bits_raw.size > 0 else None
        symbols = symbols_raw if symbols_raw.size > 0 else None

        record = cls(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            symbol_rate_hz=symbol_rate_hz,
            samples_per_symbol=samples_per_symbol,
            bits=bits,
            symbols=symbols,
            metadata=metadata_document.get("metadata", {}),
        )

        record.validate()
        return record


def load_step01_record() -> SignalRecord:
    """Adım 01 dosyalarını okuyarak ilk SignalRecord nesnesini oluşturur."""

    required_paths = {
        "bits": STEP01_DIRECTORY / "bits.npy",
        "symbols": STEP01_DIRECTORY / "symbols.npy",
        "rrc_taps": STEP01_DIRECTORY / "rrc_filter_taps.npy",
        "signal": STEP01_DIRECTORY / "clean_qpsk_signal.npy",
        "metadata": STEP01_DIRECTORY / "metadata.json",
    }

    missing = [
        str(path)
        for path in required_paths.values()
        if not path.exists()
    ]

    if missing:
        missing_text = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Adım 01 çıktılarından bazıları bulunamadı:\n"
            f"{missing_text}\n"
            "Önce şu komutu çalıştır:\n"
            "python 01_qpsk_generator.py"
        )

    bits = np.load(required_paths["bits"], allow_pickle=False)
    symbols = np.load(required_paths["symbols"], allow_pickle=False)
    rrc_taps = np.load(required_paths["rrc_taps"], allow_pickle=False)
    signal = np.load(required_paths["signal"], allow_pickle=False)

    with required_paths["metadata"].open("r", encoding="utf-8") as file:
        step01_metadata = json.load(file)

    record = SignalRecord(
        samples=np.asarray(signal, dtype=np.complex128),
        sample_rate_hz=float(step01_metadata["sample_rate_hz"]),
        symbol_rate_hz=float(step01_metadata["symbol_rate_hz"]),
        samples_per_symbol=int(step01_metadata["samples_per_symbol"]),
        bits=np.asarray(bits, dtype=np.int8),
        symbols=np.asarray(symbols, dtype=np.complex128),
        metadata={
            "source_step": 1,
            "source_description": step01_metadata.get("description"),
            "seed": step01_metadata.get("seed"),
            "rrc_roll_off": step01_metadata.get("rrc_roll_off"),
            "rrc_span_symbols": step01_metadata.get("rrc_span_symbols"),
            "rrc_number_of_taps": int(len(rrc_taps)),
            "rrc_filter_energy": float(
                np.sum(np.abs(rrc_taps) ** 2)
            ),
            "rrc_group_delay_samples": step01_metadata.get(
                "rrc_group_delay_samples"
            ),
        },
    )

    record.validate()
    return record


def verify_round_trip(
    original: SignalRecord,
    restored: SignalRecord,
) -> None:
    """Kaydet-yükle işleminin hiçbir veri kaybı oluşturmadığını doğrular."""

    if not np.array_equal(original.samples, restored.samples):
        raise RuntimeError("Round-trip testinde samples değişti.")

    if original.bits is None or restored.bits is None:
        raise RuntimeError("Round-trip testinde bits kayboldu.")

    if original.symbols is None or restored.symbols is None:
        raise RuntimeError("Round-trip testinde symbols kayboldu.")

    if not np.array_equal(original.bits, restored.bits):
        raise RuntimeError("Round-trip testinde bits değişti.")

    if not np.array_equal(original.symbols, restored.symbols):
        raise RuntimeError("Round-trip testinde symbols değişti.")

    if original.sample_rate_hz != restored.sample_rate_hz:
        raise RuntimeError("Round-trip testinde sample_rate_hz değişti.")

    if original.symbol_rate_hz != restored.symbol_rate_hz:
        raise RuntimeError("Round-trip testinde symbol_rate_hz değişti.")

    if original.samples_per_symbol != restored.samples_per_symbol:
        raise RuntimeError("Round-trip testinde samples_per_symbol değişti.")


def main() -> None:
    STEP02_DIRECTORY.mkdir(parents=True, exist_ok=True)

    original_record = load_step01_record()

    npz_path, metadata_path = original_record.save(STEP02_DIRECTORY)

    restored_record = SignalRecord.load(
        npz_path=npz_path,
        metadata_path=metadata_path,
    )

    verify_round_trip(
        original=original_record,
        restored=restored_record,
    )

    summary = restored_record.summary()

    print()
    print("=" * 66)
    print("GENESIS-DSP — ADIM 02 BAŞARIYLA TAMAMLANDI")
    print("=" * 66)
    print(f"Örnek sayısı              : {summary['number_of_samples']}")
    print(f"Sembol sayısı             : {summary['number_of_symbols']}")
    print(f"Bit çifti sayısı          : {summary['number_of_bit_pairs']}")
    print(f"Örnekleme frekansı        : {summary['sample_rate_hz']:,.0f} Hz")
    print(f"Sembol frekansı           : {summary['symbol_rate_hz']:,.0f} baud")
    print(f"Sembol başına örnek       : {summary['samples_per_symbol']}")
    print(f"Sinyal süresi             : {summary['duration_seconds']:.9f} s")
    print(f"Ortalama sinyal gücü      : {summary['average_power']:.12f}")
    print(f"Tepe genlik               : {summary['peak_magnitude']:.12f}")
    print(f"Crest factor              : {summary['crest_factor_db']:.6f} dB")
    print(f"SHA-256                   : {summary['samples_sha256']}")
    print()
    print(f"Standart veri paketi      : {npz_path}")
    print(f"Standart metadata         : {metadata_path}")
    print("Round-trip bütünlük testi : BAŞARILI")
    print("=" * 66)


if __name__ == "__main__":
    main()
