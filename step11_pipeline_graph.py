"""
GENESIS-DSP — Adım 11
DSP pipeline graph motoru.

Bu program:
1. Adım 10 BlockRegistry sistemini kullanır.
2. DSP bloklarını düğüm ve kenarlardan oluşan yönlü çevrimsiz grafikte tanımlar.
3. Graph doğrulaması, cycle detection ve topological sort yapar.
4. Tek girişten dallanan DSP pipeline'larını çalıştırır.
5. Her düğümün SignalFrame çıktısını ve yürütme kaydını saklar.
6. JSON graph config, execution plan ve test raporu üretir.

Not:
Bu sürümde her DSP düğümü tek girişlidir. Bir düğümün birden fazla
öncülü olamaz. Dallanma desteklenir; birleştirme blokları sonraki
adımlarda ayrıca eklenecektir.

Çalıştırma:
    python step11_pipeline_graph.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from step09_dsp_block_interface import (
    BlockExecutionRecord,
    SignalFrame,
    execute_block,
)
from step10_block_registry import (
    BlockRegistry,
    build_default_registry,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = BASE_DIRECTORY / "outputs" / "step11"


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    block_id: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str


@dataclass(frozen=True)
class NodeExecutionResult:
    node_id: str
    predecessor_node_id: str | None
    record: BlockExecutionRecord


@dataclass
class GraphExecutionResult:
    node_outputs: dict[str, SignalFrame]
    node_records: dict[str, NodeExecutionResult]
    topological_order: list[str]
    root_nodes: list[str]
    leaf_nodes: list[str]

    def output(self, node_id: str) -> SignalFrame:
        if node_id not in self.node_outputs:
            raise KeyError(
                f"'{node_id}' düğümünün çıktısı bulunamadı."
            )

        return self.node_outputs[node_id]


class PipelineGraph:
    """Tek girişli DSP bloklarından oluşan yönlü çevrimsiz graph."""

    def __init__(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)

        self._node_map: dict[str, GraphNode] = {}
        self._predecessors: dict[str, list[str]] = {}
        self._successors: dict[str, list[str]] = {}
        self._topological_order: list[str] = []

        self.validate()

    @classmethod
    def from_config(
        cls,
        configuration: dict[str, Any],
    ) -> "PipelineGraph":
        if not isinstance(configuration, dict):
            raise TypeError(
                "Graph yapılandırması dict olmalıdır."
            )

        allowed_keys = {
            "schema_name",
            "schema_version",
            "nodes",
            "edges",
        }
        unknown_keys = set(configuration) - allowed_keys

        if unknown_keys:
            raise ValueError(
                "Bilinmeyen graph alanları: "
                + ", ".join(sorted(unknown_keys))
            )

        raw_nodes = configuration.get("nodes")
        raw_edges = configuration.get("edges")

        if not isinstance(raw_nodes, list):
            raise TypeError("'nodes' alanı liste olmalıdır.")

        if not isinstance(raw_edges, list):
            raise TypeError("'edges' alanı liste olmalıdır.")

        nodes: list[GraphNode] = []

        for index, raw_node in enumerate(raw_nodes):
            if not isinstance(raw_node, dict):
                raise TypeError(
                    f"nodes[{index}] dict olmalıdır."
                )

            allowed_node_keys = {
                "node_id",
                "block_id",
                "parameters",
            }
            unknown_node_keys = (
                set(raw_node) - allowed_node_keys
            )

            if unknown_node_keys:
                raise ValueError(
                    f"nodes[{index}] bilinmeyen alanlar içeriyor: "
                    + ", ".join(sorted(unknown_node_keys))
                )

            if "node_id" not in raw_node:
                raise ValueError(
                    f"nodes[{index}] içinde node_id bulunmalıdır."
                )

            if "block_id" not in raw_node:
                raise ValueError(
                    f"nodes[{index}] içinde block_id bulunmalıdır."
                )

            parameters = raw_node.get("parameters", {})

            if not isinstance(parameters, dict):
                raise TypeError(
                    f"nodes[{index}].parameters dict olmalıdır."
                )

            nodes.append(
                GraphNode(
                    node_id=str(raw_node["node_id"]),
                    block_id=str(raw_node["block_id"]),
                    parameters=dict(parameters),
                )
            )

        edges: list[GraphEdge] = []

        for index, raw_edge in enumerate(raw_edges):
            if not isinstance(raw_edge, dict):
                raise TypeError(
                    f"edges[{index}] dict olmalıdır."
                )

            if set(raw_edge) != {"source", "target"}:
                raise ValueError(
                    f"edges[{index}] yalnızca source ve target "
                    "alanlarını içermelidir."
                )

            edges.append(
                GraphEdge(
                    source=str(raw_edge["source"]),
                    target=str(raw_edge["target"]),
                )
            )

        return cls(
            nodes=nodes,
            edges=edges,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "schema_name": "GENESIS-DSP PipelineGraph",
            "schema_version": "1.0.0",
            "nodes": [
                asdict(node)
                for node in self.nodes
            ],
            "edges": [
                asdict(edge)
                for edge in self.edges
            ],
        }

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError(
                "Pipeline graph en az bir düğüm içermelidir."
            )

        node_map: dict[str, GraphNode] = {}

        for node in self.nodes:
            normalized_id = node.node_id.strip()

            if not normalized_id:
                raise ValueError("node_id boş olamaz.")

            if normalized_id != node.node_id:
                raise ValueError(
                    f"node_id başında/sonunda boşluk içeremez: "
                    f"'{node.node_id}'"
                )

            if normalized_id in node_map:
                raise ValueError(
                    f"Duplicate node_id: '{normalized_id}'"
                )

            if not node.block_id.strip():
                raise ValueError(
                    f"'{normalized_id}' düğümünün block_id alanı boş."
                )

            node_map[normalized_id] = node

        predecessors = {
            node_id: []
            for node_id in node_map
        }
        successors = {
            node_id: []
            for node_id in node_map
        }

        edge_pairs: set[tuple[str, str]] = set()

        for edge in self.edges:
            if edge.source not in node_map:
                raise ValueError(
                    f"Kenar kaynağı bulunamadı: '{edge.source}'"
                )

            if edge.target not in node_map:
                raise ValueError(
                    f"Kenar hedefi bulunamadı: '{edge.target}'"
                )

            if edge.source == edge.target:
                raise ValueError(
                    f"Self-loop yasaktır: '{edge.source}'"
                )

            pair = (edge.source, edge.target)

            if pair in edge_pairs:
                raise ValueError(
                    f"Duplicate edge: {edge.source} -> {edge.target}"
                )

            edge_pairs.add(pair)
            successors[edge.source].append(edge.target)
            predecessors[edge.target].append(edge.source)

        for node_id, node_predecessors in predecessors.items():
            if len(node_predecessors) > 1:
                raise ValueError(
                    f"'{node_id}' düğümünün birden fazla girişi var. "
                    "Bu sürüm yalnızca tek girişli DSP bloklarını destekler."
                )

        for node_id in successors:
            successors[node_id].sort()

        for node_id in predecessors:
            predecessors[node_id].sort()

        topological_order = self._calculate_topological_order(
            node_map=node_map,
            predecessors=predecessors,
            successors=successors,
        )

        self._node_map = node_map
        self._predecessors = predecessors
        self._successors = successors
        self._topological_order = topological_order

    @staticmethod
    def _calculate_topological_order(
        node_map: dict[str, GraphNode],
        predecessors: dict[str, list[str]],
        successors: dict[str, list[str]],
    ) -> list[str]:
        indegrees = {
            node_id: len(predecessors[node_id])
            for node_id in node_map
        }

        ready = sorted(
            node_id
            for node_id, degree in indegrees.items()
            if degree == 0
        )

        order: list[str] = []

        while ready:
            current = ready.pop(0)
            order.append(current)

            for successor in successors[current]:
                indegrees[successor] -= 1

                if indegrees[successor] == 0:
                    ready.append(successor)
                    ready.sort()

        if len(order) != len(node_map):
            cyclic_nodes = sorted(
                node_id
                for node_id, degree in indegrees.items()
                if degree > 0
            )

            raise ValueError(
                "Pipeline graph cycle içeriyor. "
                "Cycle içindeki düğümler: "
                + ", ".join(cyclic_nodes)
            )

        return order

    @property
    def topological_order(self) -> list[str]:
        return list(self._topological_order)

    @property
    def root_nodes(self) -> list[str]:
        return sorted(
            node_id
            for node_id, predecessors
            in self._predecessors.items()
            if not predecessors
        )

    @property
    def leaf_nodes(self) -> list[str]:
        return sorted(
            node_id
            for node_id, successors
            in self._successors.items()
            if not successors
        )

    def predecessor(
        self,
        node_id: str,
    ) -> str | None:
        if node_id not in self._node_map:
            raise KeyError(
                f"'{node_id}' düğümü bulunamadı."
            )

        predecessors = self._predecessors[node_id]

        if not predecessors:
            return None

        return predecessors[0]

    def execution_plan(self) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = []

        for order_index, node_id in enumerate(
            self._topological_order
        ):
            node = self._node_map[node_id]

            plan.append(
                {
                    "order": order_index,
                    "node_id": node_id,
                    "block_id": node.block_id,
                    "parameters": node.parameters,
                    "predecessor": self.predecessor(
                        node_id
                    ),
                    "successors": list(
                        self._successors[node_id]
                    ),
                }
            )

        return plan

    def execute(
        self,
        registry: BlockRegistry,
        input_frame: SignalFrame,
    ) -> GraphExecutionResult:
        input_frame.validate()

        outputs: dict[str, SignalFrame] = {}
        records: dict[str, NodeExecutionResult] = {}

        for node_id in self._topological_order:
            node = self._node_map[node_id]
            predecessor_node_id = self.predecessor(
                node_id
            )

            if predecessor_node_id is None:
                node_input = input_frame
            else:
                node_input = outputs[
                    predecessor_node_id
                ]

            block = registry.create(
                node.block_id,
                **node.parameters,
            )

            try:
                node_output, execution_record = (
                    execute_block(
                        block=block,
                        frame=node_input,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    f"Graph düğümü çalıştırılamadı: "
                    f"node_id='{node_id}', "
                    f"block_id='{node.block_id}'"
                ) from error

            node_output.metadata = {
                **node_output.metadata,
                "graph_node_id": node_id,
                "graph_block_id": node.block_id,
                "graph_predecessor": (
                    predecessor_node_id
                ),
            }

            outputs[node_id] = node_output
            records[node_id] = NodeExecutionResult(
                node_id=node_id,
                predecessor_node_id=(
                    predecessor_node_id
                ),
                record=execution_record,
            )

        return GraphExecutionResult(
            node_outputs=outputs,
            node_records=records,
            topological_order=self.topological_order,
            root_nodes=self.root_nodes,
            leaf_nodes=self.leaf_nodes,
        )


def build_self_test_graph() -> PipelineGraph:
    configuration = {
        "schema_name": "GENESIS-DSP PipelineGraph",
        "schema_version": "1.0.0",
        "nodes": [
            {
                "node_id": "remove_dc",
                "block_id": "dc_removal",
                "parameters": {},
            },
            {
                "node_id": "downconvert",
                "block_id": "frequency_shift",
                "parameters": {
                    "frequency_hz": -62_500.0,
                    "initial_phase_degrees": 0.0,
                },
            },
            {
                "node_id": "constant_output",
                "block_id": "complex_gain",
                "parameters": {
                    "gain_real": 0.50,
                    "gain_imag": 0.25,
                },
            },
            {
                "node_id": "tone_branch",
                "block_id": "complex_gain",
                "parameters": {
                    "gain_real": 0.25,
                    "gain_imag": -0.50,
                },
            },
        ],
        "edges": [
            {
                "source": "remove_dc",
                "target": "downconvert",
            },
            {
                "source": "downconvert",
                "target": "constant_output",
            },
            {
                "source": "remove_dc",
                "target": "tone_branch",
            },
        ],
    }

    return PipelineGraph.from_config(
        configuration
    )


def run_graph_self_test(
    registry: BlockRegistry,
) -> dict[str, Any]:
    sample_rate_hz = 1_000_000.0
    number_of_samples = 16_384
    tone_frequency_hz = 62_500.0
    injected_dc = complex(0.18, -0.09)

    sample_indices = np.arange(
        number_of_samples,
        dtype=np.float64,
    )

    tone = np.exp(
        1j
        * 2.0
        * np.pi
        * tone_frequency_hz
        * sample_indices
        / sample_rate_hz
    )

    input_frame = SignalFrame(
        samples=(
            tone + injected_dc
        ).astype(np.complex128),
        sample_rate_hz=sample_rate_hz,
        metadata={
            "source": "step11_self_test",
            "tone_frequency_hz": (
                tone_frequency_hz
            ),
        },
    )
    input_frame.validate()

    graph = build_self_test_graph()
    result = graph.execute(
        registry=registry,
        input_frame=input_frame,
    )

    constant_output = result.output(
        "constant_output"
    )
    tone_branch = result.output(
        "tone_branch"
    )

    expected_constant = np.full(
        number_of_samples,
        complex(0.50, 0.25),
        dtype=np.complex128,
    )

    constant_maximum_error = float(
        np.max(
            np.abs(
                constant_output.samples
                - expected_constant
            )
        )
    )

    expected_tone_branch = (
        tone * complex(0.25, -0.50)
    ).astype(np.complex128)

    tone_branch_maximum_error = float(
        np.max(
            np.abs(
                tone_branch.samples
                - expected_tone_branch
            )
        )
    )

    if constant_maximum_error > 1e-10:
        raise RuntimeError(
            "Sabit çıkış dalı öz testi başarısız."
        )

    if tone_branch_maximum_error > 1e-10:
        raise RuntimeError(
            "Tone dalı öz testi başarısız."
        )

    expected_order = [
        "remove_dc",
        "downconvert",
        "constant_output",
        "tone_branch",
    ]

    if result.topological_order != expected_order:
        raise RuntimeError(
            "Topological order beklenen sırayla eşleşmedi."
        )

    if result.root_nodes != ["remove_dc"]:
        raise RuntimeError(
            "Root node tespiti hatalı."
        )

    if result.leaf_nodes != [
        "constant_output",
        "tone_branch",
    ]:
        raise RuntimeError(
            "Leaf node tespiti hatalı."
        )

    if len(result.node_records) != 4:
        raise RuntimeError(
            "Node execution record sayısı hatalı."
        )

    return {
        "status": "PASSED",
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "root_nodes": result.root_nodes,
        "leaf_nodes": result.leaf_nodes,
        "topological_order": (
            result.topological_order
        ),
        "constant_branch_maximum_error": (
            constant_maximum_error
        ),
        "tone_branch_maximum_error": (
            tone_branch_maximum_error
        ),
        "execution_records": {
            node_id: {
                "node_id": node_result.node_id,
                "predecessor_node_id": (
                    node_result.predecessor_node_id
                ),
                "record": asdict(
                    node_result.record
                ),
            }
            for node_id, node_result
            in result.node_records.items()
        },
    }


def run_validation_self_tests() -> dict[str, str]:
    tests: dict[str, str] = {}

    try:
        PipelineGraph(
            nodes=[
                GraphNode(
                    node_id="a",
                    block_id="dc_removal",
                    parameters={},
                ),
                GraphNode(
                    node_id="b",
                    block_id="complex_gain",
                    parameters={},
                ),
            ],
            edges=[
                GraphEdge(
                    source="a",
                    target="b",
                ),
                GraphEdge(
                    source="b",
                    target="a",
                ),
            ],
        )
    except ValueError as error:
        if "cycle" not in str(error).lower():
            raise
        tests["cycle_rejection"] = "PASSED"
    else:
        raise RuntimeError(
            "Cycle içeren graph reddedilmedi."
        )

    try:
        PipelineGraph(
            nodes=[
                GraphNode(
                    node_id="a",
                    block_id="dc_removal",
                    parameters={},
                ),
                GraphNode(
                    node_id="b",
                    block_id="complex_gain",
                    parameters={},
                ),
                GraphNode(
                    node_id="c",
                    block_id="frequency_shift",
                    parameters={},
                ),
            ],
            edges=[
                GraphEdge(
                    source="a",
                    target="c",
                ),
                GraphEdge(
                    source="b",
                    target="c",
                ),
            ],
        )
    except ValueError as error:
        if "birden fazla" not in str(error):
            raise
        tests["multiple_parent_rejection"] = (
            "PASSED"
        )
    else:
        raise RuntimeError(
            "Birden fazla girişli node reddedilmedi."
        )

    try:
        PipelineGraph(
            nodes=[
                GraphNode(
                    node_id="duplicate",
                    block_id="dc_removal",
                    parameters={},
                ),
                GraphNode(
                    node_id="duplicate",
                    block_id="complex_gain",
                    parameters={},
                ),
            ],
            edges=[],
        )
    except ValueError as error:
        if "duplicate" not in str(error).lower():
            raise
        tests["duplicate_node_rejection"] = (
            "PASSED"
        )
    else:
        raise RuntimeError(
            "Duplicate node_id reddedilmedi."
        )

    try:
        PipelineGraph(
            nodes=[
                GraphNode(
                    node_id="a",
                    block_id="dc_removal",
                    parameters={},
                )
            ],
            edges=[
                GraphEdge(
                    source="a",
                    target="missing",
                )
            ],
        )
    except ValueError:
        tests["missing_node_rejection"] = (
            "PASSED"
        )
    else:
        raise RuntimeError(
            "Eksik edge node'u reddedilmedi."
        )

    return tests


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = build_default_registry()
    graph = build_self_test_graph()

    graph_self_test = run_graph_self_test(
        registry
    )
    validation_tests = (
        run_validation_self_tests()
    )

    config_path = (
        OUTPUT_DIRECTORY
        / "pipeline_graph_config.json"
    )
    plan_path = (
        OUTPUT_DIRECTORY
        / "pipeline_execution_plan.json"
    )
    report_path = (
        OUTPUT_DIRECTORY
        / "pipeline_graph_report.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            graph.to_config(),
            file,
            indent=4,
            ensure_ascii=False,
        )

    with plan_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "schema_name": (
                    "GENESIS-DSP ExecutionPlan"
                ),
                "schema_version": "1.0.0",
                "root_nodes": graph.root_nodes,
                "leaf_nodes": graph.leaf_nodes,
                "topological_order": (
                    graph.topological_order
                ),
                "plan": graph.execution_plan(),
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    report = {
        "project": "GENESIS-DSP",
        "step": 11,
        "description": (
            "Directed acyclic DSP pipeline graph engine"
        ),
        "graph_self_test": graph_self_test,
        "validation_self_tests": (
            validation_tests
        ),
    }

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

    print()
    print("=" * 74)
    print(
        "GENESIS-DSP — ADIM 11 BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 74)
    print(
        f"Graph düğüm sayısı          : "
        f"{graph_self_test['node_count']}"
    )
    print(
        f"Graph kenar sayısı          : "
        f"{graph_self_test['edge_count']}"
    )
    print(
        "Topological sort            : BAŞARILI"
    )
    print(
        "Dallanmış graph yürütmesi   : BAŞARILI"
    )
    print(
        "Cycle koruması              : BAŞARILI"
    )
    print(
        "Çoklu giriş koruması        : BAŞARILI"
    )
    print(
        f"Sabit dal maksimum hatası   : "
        f"{graph_self_test['constant_branch_maximum_error']:.3e}"
    )
    print(
        f"Tone dal maksimum hatası    : "
        f"{graph_self_test['tone_branch_maximum_error']:.3e}"
    )
    print(
        f"Graph config                : "
        f"{config_path}"
    )
    print(
        f"Execution plan              : "
        f"{plan_path}"
    )
    print(
        f"Test raporu                 : "
        f"{report_path}"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
