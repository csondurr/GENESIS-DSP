
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIRECTORY / "outputs" / "step09"

ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    parameter_type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def validate_value(self, value: Any) -> None:
        if self.parameter_type == "float":
            if not isinstance(value, (int, float, np.number)):
                raise TypeError(f"{self.name} sayısal olmalıdır.")

            numeric_value = float(value)

            if not np.isfinite(numeric_value):
                raise ValueError(f"{self.name} sonlu olmalıdır.")

            if self.minimum is not None and numeric_value < self.minimum:
                raise ValueError(
                    f"{self.name} en az {self.minimum} olmalıdır."
                )

            if self.maximum is not None and numeric_value > self.maximum:
                raise ValueError(
                    f"{self.name} en fazla {self.maximum} olmalıdır."
                )

        elif self.parameter_type == "int":
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{self.name} tam sayı olmalıdır.")

            integer_value = int(value)

            if self.minimum is not None and integer_value < self.minimum:
                raise ValueError(
                    f"{self.name} en az {self.minimum} olmalıdır."
                )

            if self.maximum is not None and integer_value > self.maximum:
                raise ValueError(
                    f"{self.name} en fazla {self.maximum} olmalıdır."
                )

        elif self.parameter_type == "bool":
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{self.name} boolean olmalıdır.")

        elif self.parameter_type == "complex":
            try:
                complex(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"{self.name} kompleks sayıya çevrilebilmelidir."
                ) from error

        else:
            raise ValueError(
                f"Desteklenmeyen parametre türü: {self.parameter_type}"
            )


@dataclass
class SignalFrame:
    samples: ComplexArray
    sample_rate_hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.samples = np.asarray(
            self.samples,
            dtype=np.complex128,
        )

        if self.samples.ndim != 1:
            raise ValueError("samples tek boyutlu olmalıdır.")

        if len(self.samples) == 0:
            raise ValueError("samples boş olamaz.")

        if not np.all(np.isfinite(self.samples.real)):
            raise ValueError(
                "samples gerçek kısmında NaN veya Inf bulundu."
            )

        if not np.all(np.isfinite(self.samples.imag)):
            raise ValueError(
                "samples sanal kısmında NaN veya Inf bulundu."
            )

        if (
            not np.isfinite(self.sample_rate_hz)
            or self.sample_rate_hz <= 0.0
        ):
            raise ValueError(
                "sample_rate_hz pozitif ve sonlu olmalıdır."
            )

    @property
    def number_of_samples(self) -> int:
        return int(len(self.samples))

    @property
    def average_power(self) -> float:
        return float(
            np.mean(np.abs(self.samples) ** 2)
        )

    @property
    def duration_seconds(self) -> float:
        return float(
            self.number_of_samples / self.sample_rate_hz
        )

    def with_samples(
        self,
        samples: ComplexArray,
        metadata_update: dict[str, Any] | None = None,
    ) -> "SignalFrame":
        new_metadata = dict(self.metadata)

        if metadata_update:
            new_metadata.update(metadata_update)

        frame = SignalFrame(
            samples=np.asarray(
                samples,
                dtype=np.complex128,
            ),
            sample_rate_hz=self.sample_rate_hz,
            metadata=new_metadata,
        )
        frame.validate()
        return frame


@dataclass(frozen=True)
class BlockExecutionRecord:
    block_id: str
    block_name: str
    configuration: dict[str, Any]
    input_samples: int
    output_samples: int
    input_power: float
    output_power: float
    elapsed_milliseconds: float
    estimated_mac_count: int


class DSPBlock(ABC):
    block_id: str = "abstract"
    block_name: str = "Abstract DSP Block"
    category: str = "abstract"

    def __init__(self, **parameters: Any) -> None:
        specifications = {
            spec.name: spec
            for spec in self.parameter_specs()
        }

        unknown_parameters = set(parameters) - set(specifications)

        if unknown_parameters:
            raise ValueError(
                "Bilinmeyen parametreler: "
                + ", ".join(sorted(unknown_parameters))
            )

        resolved_parameters: dict[str, Any] = {}

        for name, specification in specifications.items():
            value = parameters.get(
                name,
                specification.default,
            )
            specification.validate_value(value)
            resolved_parameters[name] = value

        self.parameters = resolved_parameters

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return ()

    def configuration(self) -> dict[str, Any]:
        return dict(self.parameters)

    def describe(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_name": self.block_name,
            "category": self.category,
            "parameters": [
                asdict(spec)
                for spec in self.parameter_specs()
            ],
            "configuration": self.configuration(),
        }

    def validate_input(self, frame: SignalFrame) -> None:
        frame.validate()

    @abstractmethod
    def process(self, frame: SignalFrame) -> SignalFrame:
        raise NotImplementedError

    def estimated_mac_count(
        self,
        input_samples: int,
    ) -> int:
        return int(input_samples)


def execute_block(
    block: DSPBlock,
    frame: SignalFrame,
) -> tuple[SignalFrame, BlockExecutionRecord]:
    block.validate_input(frame)

    start_time = time.perf_counter()
    output = block.process(frame)
    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1_000.0

    output.validate()

    record = BlockExecutionRecord(
        block_id=block.block_id,
        block_name=block.block_name,
        configuration=block.configuration(),
        input_samples=frame.number_of_samples,
        output_samples=output.number_of_samples,
        input_power=frame.average_power,
        output_power=output.average_power,
        elapsed_milliseconds=float(elapsed_ms),
        estimated_mac_count=block.estimated_mac_count(
            frame.number_of_samples
        ),
    )

    return output, record


class DCRemovalBlock(DSPBlock):
    block_id = "dc_removal"
    block_name = "DC Removal"
    category = "correction"

    def process(self, frame: SignalFrame) -> SignalFrame:
        estimated_dc = complex(
            np.mean(frame.samples)
        )

        corrected = frame.samples - estimated_dc

        return frame.with_samples(
            corrected,
            {
                "estimated_dc_real": float(
                    estimated_dc.real
                ),
                "estimated_dc_imag": float(
                    estimated_dc.imag
                ),
            },
        )


class ComplexGainBlock(DSPBlock):
    block_id = "complex_gain"
    block_name = "Complex Gain"
    category = "amplitude_phase"

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="gain_real",
                parameter_type="float",
                default=1.0,
                minimum=-1000.0,
                maximum=1000.0,
                description="Kompleks kazancın gerçek kısmı.",
            ),
            ParameterSpec(
                name="gain_imag",
                parameter_type="float",
                default=0.0,
                minimum=-1000.0,
                maximum=1000.0,
                description="Kompleks kazancın sanal kısmı.",
            ),
        )

    def process(self, frame: SignalFrame) -> SignalFrame:
        gain = complex(
            float(self.parameters["gain_real"]),
            float(self.parameters["gain_imag"]),
        )

        output = frame.samples * gain

        return frame.with_samples(
            output,
            {
                "complex_gain_real": float(
                    gain.real
                ),
                "complex_gain_imag": float(
                    gain.imag
                ),
            },
        )


class FrequencyShiftBlock(DSPBlock):
    block_id = "frequency_shift"
    block_name = "Frequency Shift"
    category = "frequency"

    @classmethod
    def parameter_specs(cls) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="frequency_hz",
                parameter_type="float",
                default=0.0,
                description=(
                    "Pozitif değer yukarı, negatif değer "
                    "aşağı frekans kaydırır."
                ),
            ),
            ParameterSpec(
                name="initial_phase_degrees",
                parameter_type="float",
                default=0.0,
                minimum=-360.0,
                maximum=360.0,
                description="Başlangıç fazı.",
            ),
        )

    def validate_input(self, frame: SignalFrame) -> None:
        super().validate_input(frame)

        frequency_hz = abs(
            float(self.parameters["frequency_hz"])
        )

        if frequency_hz >= frame.sample_rate_hz / 2.0:
            raise ValueError(
                "Frekans kayması Nyquist frekansından küçük olmalıdır."
            )

    def process(self, frame: SignalFrame) -> SignalFrame:
        frequency_hz = float(
            self.parameters["frequency_hz"]
        )
        initial_phase = np.deg2rad(
            float(
                self.parameters[
                    "initial_phase_degrees"
                ]
            )
        )

        indices = np.arange(
            frame.number_of_samples,
            dtype=np.float64,
        )

        phase = (
            2.0
            * np.pi
            * frequency_hz
            * indices
            / frame.sample_rate_hz
            + initial_phase
        )

        output = frame.samples * np.exp(1j * phase)

        return frame.with_samples(
            output,
            {
                "frequency_shift_hz": frequency_hz,
                "initial_phase_degrees": float(
                    self.parameters[
                        "initial_phase_degrees"
                    ]
                ),
            },
        )


def run_self_test() -> dict[str, Any]:
    sample_rate_hz = 1_000_000.0
    number_of_samples = 8192
    tone_frequency_hz = 40_000.0
    dc_offset = complex(0.20, -0.12)

    indices = np.arange(
        number_of_samples,
        dtype=np.float64,
    )

    tone = np.exp(
        1j
        * 2.0
        * np.pi
        * tone_frequency_hz
        * indices
        / sample_rate_hz
    )

    input_samples = (
        tone + dc_offset
    ).astype(np.complex128)

    frame = SignalFrame(
        samples=input_samples,
        sample_rate_hz=sample_rate_hz,
        metadata={
            "self_test": True,
            "tone_frequency_hz": tone_frequency_hz,
        },
    )
    frame.validate()

    blocks: list[DSPBlock] = [
        DCRemovalBlock(),
        ComplexGainBlock(
            gain_real=0.5,
            gain_imag=0.25,
        ),
        FrequencyShiftBlock(
            frequency_hz=-tone_frequency_hz,
            initial_phase_degrees=0.0,
        ),
    ]

    execution_records: list[
        BlockExecutionRecord
    ] = []

    current_frame = frame

    for block in blocks:
        current_frame, record = execute_block(
            block,
            current_frame,
        )
        execution_records.append(record)

    residual_dc_after_first_block = complex(
        np.mean(
            input_samples
            - np.mean(input_samples)
        )
    )

    final_mean = complex(
        np.mean(current_frame.samples)
    )

    if abs(residual_dc_after_first_block) > 1e-12:
        raise RuntimeError(
            "DC removal öz testi başarısız."
        )

    expected_final_magnitude = abs(
        complex(0.5, 0.25)
    )

    if not np.isclose(
        abs(final_mean),
        expected_final_magnitude,
        atol=2e-3,
    ):
        raise RuntimeError(
            "Gain/frequency shift öz testi başarısız."
        )

    return {
        "status": "PASSED",
        "input": {
            "number_of_samples": (
                frame.number_of_samples
            ),
            "sample_rate_hz": (
                frame.sample_rate_hz
            ),
            "average_power": (
                frame.average_power
            ),
        },
        "final": {
            "number_of_samples": (
                current_frame.number_of_samples
            ),
            "average_power": (
                current_frame.average_power
            ),
            "mean_real": float(
                final_mean.real
            ),
            "mean_imag": float(
                final_mean.imag
            ),
        },
        "blocks": [
            block.describe()
            for block in blocks
        ],
        "execution_records": [
            asdict(record)
            for record in execution_records
        ],
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "project": "GENESIS-DSP",
        "step": 9,
        "description": (
            "Common DSP block interface and execution contract"
        ),
        "self_test": run_self_test(),
    }

    report_path = (
        OUTPUT_DIRECTORY
        / "dsp_block_interface_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    records = report[
        "self_test"
    ]["execution_records"]

    print()
    print("=" * 70)
    print(
        "GENESIS-DSP — ADIM 09 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 70)
    print(
        f"Tanımlanan örnek blok sayısı : "
        f"{len(records)}"
    )
    print(
        "SignalFrame doğrulaması      : BAŞARILI"
    )
    print(
        "Parametre şeması             : BAŞARILI"
    )
    print(
        "Block yürütme kayıtları      : BAŞARILI"
    )
    print(
        "DC/Gain/Frequency öz testi   : BAŞARILI"
    )
    print(
        f"Rapor                        : "
        f"{report_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
