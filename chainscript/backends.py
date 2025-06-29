"""
ChainScript Backends: Multi-QPU Provider Support
===============================================

Manage connections to multiple quantum computing providers
including Rigetti, IBM, IonQ, and simulators.
"""

from typing import List, Dict, Any
from enum import Enum


class QPUProvider(Enum):
    RIGETTI = "rigetti"
    IBM = "ibm"
    IONQ = "ionq"
    GOOGLE = "google"
    SIMULATOR = "simulator"


def list_available_backends() -> List[Dict[str, Any]]:
    """List all available quantum backends"""
    return [
        {
            "provider": QPUProvider.RIGETTI.value,
            "backends": ["Aspen-M-3", "2q-qvm", "9q-qvm"],
            "status": "available",
        },
        {
            "provider": QPUProvider.IBM.value,
            "backends": ["ibm_nairobi", "ibm_osaka", "simulator_mps"],
            "status": "integration_pending",
        },
        {
            "provider": QPUProvider.IONQ.value,
            "backends": ["ionq_aria", "ionq_forte"],
            "status": "integration_pending",
        },
        {
            "provider": QPUProvider.SIMULATOR.value,
            "backends": ["qvm", "wavefunction"],
            "status": "available",
        },
    ]


def get_backend_capabilities(provider: str, backend: str) -> Dict[str, Any]:
    """Get specific backend capabilities"""
    capabilities = {
        "rigetti": {
            "max_qubits": 80,
            "connectivity": "heavy_hex",
            "gate_fidelity": 0.99,
            "readout_fidelity": 0.95,
        },
        "ibm": {
            "max_qubits": 127,
            "connectivity": "heavy_hex",
            "gate_fidelity": 0.999,
            "readout_fidelity": 0.97,
        },
        "simulator": {
            "max_qubits": 32,
            "connectivity": "all_to_all",
            "gate_fidelity": 1.0,
            "readout_fidelity": 1.0,
        },
    }

    return capabilities.get(provider, {"status": "unknown"})
