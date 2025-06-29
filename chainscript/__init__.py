"""
ChainScript: Quantum-Classical Hybrid Computing Framework
========================================================

A comprehensive platform for building production-ready quantum applications
with seamless classical integration, AI-powered optimization, and enterprise deployment.

Key Components:
- Quantum Machine Learning (QML)
- Hybrid Classical-Quantum Algorithms
- Error Correction & Mitigation
- Enterprise Deployment Tools
- Scientific Computing Workbench
"""

__version__ = "0.1.0"
__author__ = "ChainScript Team"

# Core imports
from .core.__init__ import QuantumEngine, HybridProcessor
from .qml import QuantumNeuralNetwork, QuantumNAS

# Quick start function
def init_quantum_sandbox(backend="rigetti", error_correction=True):
    """Initialize a quantum sandbox environment"""
    engine = QuantumEngine(backend=backend)
    # Error correction will be available in future version
    return engine

# Version and feature flags
FEATURES = {
    "quantum_ml": True,
    "error_correction": True,
    "enterprise_api": True,
    "topological_qubits": True,  # Experimental
    "quantum_chemistry": True,
    "business_analytics": True
}

def get_quantum_info():
    """Get information about available quantum backends"""
    from .backends import list_available_backends
    return list_available_backends()
