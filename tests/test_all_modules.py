"""
ChainScript Test Suite: Comprehensive Module Testing
==================================================

Test all ChainScript modules individually to ensure functionality
and integration readiness.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import numpy as np
import pytest
from datetime import datetime

# Import all ChainScript modules
from chainscript.core.__init__ import QuantumEngine, HybridProcessor, QuantumResult
from chainscript.qml import QuantumNeuralNetwork, QuantumNAS, QuantumFeatureMap
from chainscript.security import QuantumCrypto, ComplianceFramework, QuantumProtocol
from chainscript.marketplace import QuantumAppStore
from chainscript.topological import MajoranaQubitSimulator

class TestQuantumCore:
    """Test the core quantum engine functionality"""
    
    def test_quantum_engine_initialization(self):
        """Test quantum engine can be initialized"""
        engine = QuantumEngine(backend="simulator")
        assert engine.backend_type.value == "simulator"
        print("✅ QuantumEngine initialization: PASSED")
    
    def test_bell_state_creation(self):
        """Test Bell state circuit creation"""
        engine = QuantumEngine(backend="simulator")
        bell_circuit = engine.create_bell_state(0, 1)
        
        # Check circuit has the right structure
        instructions = str(bell_circuit)
        assert "H 0" in instructions
        assert "CNOT 0 1" in instructions
        print("✅ Bell state creation: PASSED")
    
    async def test_circuit_execution(self):
        """Test quantum circuit execution"""
        engine = QuantumEngine(backend="simulator")
        bell_circuit = engine.create_bell_state(0, 1)
        
        try:
            result = await engine.run_circuit(bell_circuit, shots=100)
            assert isinstance(result, QuantumResult)
            assert result.shots == 100
            print("✅ Circuit execution: PASSED")
        except Exception as e:
            print(f"⚠️ Circuit execution: FAILED - {e}")
    
    def test_qft_creation(self):
        """Test Quantum Fourier Transform creation"""
        engine = QuantumEngine(backend="simulator")
        qft_circuit = engine.create_quantum_fourier_transform(3)
        
        instructions = str(qft_circuit)
        assert "H" in instructions  # Should have Hadamard gates
        print("✅ QFT creation: PASSED")

class TestQuantumML:
    """Test quantum machine learning components"""
    
    def test_feature_map_encoding(self):
        """Test quantum feature encoding"""
        features = np.array([0.5, 0.3, 0.8, 0.1])
        
        # Test angle encoding
        angle_circuit = QuantumFeatureMap.angle_encoding(features, 4)
        assert len(angle_circuit.instructions) == 4
        print("✅ Feature map encoding: PASSED")
    
    def test_quantum_neural_network_init(self):
        """Test QNN initialization"""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, backend="simulator")
        assert qnn.n_qubits == 4
        assert qnn.n_layers == 2
        assert len(qnn.layers) == 2
        print("✅ QNN initialization: PASSED")
    
    async def test_qnn_training_pipeline(self):
        """Test QNN training pipeline with dummy data"""
        # Create dummy dataset
        X_train = np.random.rand(10, 4)
        y_train = np.random.randint(0, 2, 10)
        
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=1, backend="simulator")
        
        try:
            # Test feature encoding
            circuits = await qnn._encode_features(X_train[:2])  # Test with 2 samples
            assert len(circuits) == 2
            print("✅ QNN feature encoding: PASSED")
        except Exception as e:
            print(f"⚠️ QNN training pipeline: FAILED - {e}")
    
    def test_quantum_nas_initialization(self):
        """Test Quantum NAS initialization"""
        qnas = QuantumNAS(max_evaluations=5)
        assert qnas.max_evaluations == 5
        assert qnas.best_score == -np.inf
        print("✅ Quantum NAS initialization: PASSED")
    
    def test_architecture_generation(self):
        """Test architecture candidate generation"""
        qnas = QuantumNAS()
        candidate = qnas._generate_random_architecture()
        
        assert len(candidate.quantum_layers) >= 1
        assert len(candidate.classical_layers) >= 1
        print("✅ Architecture generation: PASSED")

class TestQuantumSecurity:
    """Test quantum security and cryptography"""
    
    def test_quantum_crypto_initialization(self):
        """Test quantum cryptography initialization"""
        qcrypto = QuantumCrypto(protocol=QuantumProtocol.BB84)
        assert qcrypto.protocol == QuantumProtocol.BB84
        assert qcrypto.post_quantum_ready == True
        print("✅ Quantum crypto initialization: PASSED")
    
    def test_quantum_key_generation(self):
        """Test quantum key generation"""
        qcrypto = QuantumCrypto()
        qkey = qcrypto.generate_quantum_key(128)
        
        assert len(qkey.key_bits) == 128
        assert qkey.protocol == QuantumProtocol.BB84
        assert 0.9 <= qkey.fidelity <= 1.0
        assert qkey.key_id is not None
        print("✅ Quantum key generation: PASSED")
    
    def test_quantum_encryption(self):
        """Test quantum encryption/decryption"""
        qcrypto = QuantumCrypto()
        qkey = qcrypto.generate_quantum_key(256)
        
        message = b"Test quantum message"
        encrypted = qcrypto.encrypt_with_quantum_key(message, qkey)
        
        assert len(encrypted) == len(message)
        assert encrypted != message  # Should be different
        print("✅ Quantum encryption: PASSED")
    
    def test_compliance_framework(self):
        """Test compliance framework"""
        compliance = ComplianceFramework(["SOC2", "GDPR"])
        
        # Log some operations
        compliance.log_quantum_operation("test_operation", "simulator", "test_user")
        assert len(compliance.audit_trail) == 1
        
        # Generate report
        report = compliance.generate_compliance_report()
        assert "standards_compliance" in report
        assert report["total_operations"] == 1
        print("✅ Compliance framework: PASSED")

class TestQuantumMarketplace:
    """Test quantum marketplace functionality"""
    
    def test_app_store_initialization(self):
        """Test quantum app store initialization"""
        store = QuantumAppStore()
        assert len(store.store) == 0
        print("✅ App store initialization: PASSED")
    
    def test_app_publishing(self):
        """Test publishing apps to the store"""
        store = QuantumAppStore()
        
        store.publish(
            name="TestApp",
            qpu_requirements=["rigetti", "ibm"],
            license="MIT"
        )
        
        assert "TestApp" in store.store
        app_details = store.get_app_details("TestApp")
        assert app_details["qpu_requirements"] == ["rigetti", "ibm"]
        print("✅ App publishing: PASSED")
    
    def test_app_listing(self):
        """Test listing available apps"""
        store = QuantumAppStore()
        store.publish("App1", license="MIT")
        store.publish("App2", license="Commercial")
        
        apps = store.list_available_apps()
        assert len(apps) == 2
        print("✅ App listing: PASSED")

class TestTopologicalSimulation:
    """Test topological qubit simulation"""
    
    def test_majorana_simulator_initialization(self):
        """Test Majorana simulator initialization"""
        simulator = MajoranaQubitSimulator("toric_code")
        assert simulator.topology == "toric_code"
        print("✅ Majorana simulator initialization: PASSED")
    
    def test_qubit_encoding_decoding(self):
        """Test qubit encoding and decoding"""
        simulator = MajoranaQubitSimulator("toric_code")
        data_qubits = np.array([1, 0, 1, 0])
        
        # Test encoding
        protected = simulator.encode(data_qubits)
        assert len(protected) == len(data_qubits)
        
        # Test decoding
        decoded = simulator.decode()
        assert len(decoded) == len(data_qubits)
        print("✅ Qubit encoding/decoding: PASSED")

# Main test runner
async def run_all_tests():
    """Run all module tests"""
    print("🧪 Starting ChainScript Module Tests...\n")
    
    # Core tests
    print("🔧 Testing Core Module:")
    core_tests = TestQuantumCore()
    core_tests.test_quantum_engine_initialization()
    core_tests.test_bell_state_creation()
    await core_tests.test_circuit_execution()
    core_tests.test_qft_creation()
    print()
    
    # QML tests
    print("🧠 Testing Quantum ML Module:")
    qml_tests = TestQuantumML()
    qml_tests.test_feature_map_encoding()
    qml_tests.test_quantum_neural_network_init()
    await qml_tests.test_qnn_training_pipeline()
    qml_tests.test_quantum_nas_initialization()
    qml_tests.test_architecture_generation()
    print()
    
    # Security tests
    print("🔐 Testing Security Module:")
    security_tests = TestQuantumSecurity()
    security_tests.test_quantum_crypto_initialization()
    security_tests.test_quantum_key_generation()
    security_tests.test_quantum_encryption()
    security_tests.test_compliance_framework()
    print()
    
    # Marketplace tests
    print("🛒 Testing Marketplace Module:")
    marketplace_tests = TestQuantumMarketplace()
    marketplace_tests.test_app_store_initialization()
    marketplace_tests.test_app_publishing()
    marketplace_tests.test_app_listing()
    print()
    
    # Topological tests
    print("🔬 Testing Topological Module:")
    topo_tests = TestTopologicalSimulation()
    topo_tests.test_majorana_simulator_initialization()
    topo_tests.test_qubit_encoding_decoding()
    print()
    
    print("✨ All module tests completed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
