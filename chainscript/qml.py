"""
ChainScript QML: Quantum Machine Learning
========================================

Advanced quantum machine learning capabilities including:
- Quantum Neural Networks
- Neural Architecture Search for hybrid models
- Quantum feature maps
- Variational quantum classifiers
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
import asyncio
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import joblib

from pyquil import Program
from pyquil.gates import RX, RY, RZ, CNOT, H
from .core import QuantumEngine, QuantumResult


@dataclass
class ArchitectureCandidate:
    """Represents a neural architecture candidate"""

    quantum_layers: List[Dict]
    classical_layers: List[Dict]
    connections: List[Tuple[int, int]]
    performance: Optional[float] = None
    complexity_score: Optional[float] = None


class QuantumFeatureMap:
    """Quantum feature encoding strategies"""

    @staticmethod
    def angle_encoding(features: np.ndarray, n_qubits: int) -> Program:
        """Encode classical features into quantum states using angle encoding"""
        program = Program()

        # Normalize features to [0, 2π]
        normalized_features = (
            2 * np.pi * (features - features.min()) / (features.max() - features.min())
        )

        for i in range(min(len(normalized_features), n_qubits)):
            program += RY(normalized_features[i], i)

        return program

    @staticmethod
    def amplitude_encoding(features: np.ndarray, n_qubits: int) -> Program:
        """Encode classical features using amplitude encoding"""
        program = Program()

        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        # Create superposition based on feature amplitudes
        for i in range(min(len(features), n_qubits)):
            if features[i] != 0:
                angle = 2 * np.arcsin(np.sqrt(abs(features[i])))
                program += RY(angle, i)

        return program

    @staticmethod
    def entangling_feature_map(
        features: np.ndarray, n_qubits: int, depth: int = 2
    ) -> Program:
        """Create entangling feature map with parameterized gates"""
        program = Program()

        for d in range(depth):
            # Feature encoding layer
            for i in range(min(len(features), n_qubits)):
                program += RZ(features[i] * (d + 1), i)
                program += RY(features[i] * (d + 1), i)

            # Entangling layer
            for i in range(n_qubits - 1):
                program += CNOT(i, i + 1)

        return program


class QuantumLayer:
    """A parameterized quantum layer for neural networks"""

    def __init__(self, n_qubits: int, n_params: int, layer_type: str = "variational"):
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.layer_type = layer_type
        self.parameters = np.random.uniform(0, 2 * np.pi, n_params)

    def create_circuit(self, parameters: Optional[np.ndarray] = None) -> Program:
        """Create the quantum circuit for this layer"""
        if parameters is not None:
            params = parameters
        else:
            params = self.parameters

        program = Program()
        param_idx = 0

        if self.layer_type == "variational":
            # Variational layer with RY gates and entangling CNOTs
            for i in range(self.n_qubits):
                if param_idx < len(params):
                    program += RY(params[param_idx], i)
                    param_idx += 1

            # Entangling layer
            for i in range(self.n_qubits - 1):
                program += CNOT(i, i + 1)

        elif self.layer_type == "strongly_entangling":
            # Strongly entangling layer
            for i in range(self.n_qubits):
                if param_idx < len(params):
                    program += RX(params[param_idx], i)
                    param_idx += 1
                if param_idx < len(params):
                    program += RY(params[param_idx], i)
                    param_idx += 1
                if param_idx < len(params):
                    program += RZ(params[param_idx], i)
                    param_idx += 1

            # All-to-all entanglement
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    program += CNOT(i, j)

        return program


class QuantumNeuralNetwork(BaseEstimator, ClassifierMixin):
    """Hybrid Quantum-Classical Neural Network"""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        feature_map: str = "angle",
        backend: str = "rigetti",
        shots: int = 1024,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.backend = backend
        self.shots = shots

        self.quantum_engine = QuantumEngine(backend)
        self.layers = []
        self.classical_weights = None
        self.is_fitted = False

        # Initialize quantum layers
        for i in range(n_layers):
            layer = QuantumLayer(n_qubits, n_qubits * 3, "strongly_entangling")
            self.layers.append(layer)

    async def _encode_features(self, X: np.ndarray) -> List[Program]:
        """Encode classical features into quantum circuits"""
        circuits = []

        for sample in X:
            if self.feature_map == "angle":
                circuit = QuantumFeatureMap.angle_encoding(sample, self.n_qubits)
            elif self.feature_map == "amplitude":
                circuit = QuantumFeatureMap.amplitude_encoding(sample, self.n_qubits)
            else:
                circuit = QuantumFeatureMap.entangling_feature_map(
                    sample, self.n_qubits
                )

            circuits.append(circuit)

        return circuits

    async def _quantum_forward_pass(
        self, feature_circuits: List[Program]
    ) -> np.ndarray:
        """Execute quantum forward pass"""
        quantum_features = []

        for circuit in feature_circuits:
            # Add quantum layers
            full_circuit = circuit
            for layer in self.layers:
                full_circuit += layer.create_circuit()

            # Add measurements
            full_circuit = full_circuit.declare("ro", "BIT", self.n_qubits)
            for i in range(self.n_qubits):
                full_circuit += MEASURE(i, ("ro", i))

            # Execute circuit
            result = await self.quantum_engine.run_circuit(full_circuit, self.shots)

            # Convert measurements to features
            measurements = result.result
            expectation_values = np.mean(measurements, axis=0)
            quantum_features.append(expectation_values)

        return np.array(quantum_features)

    def _classical_forward_pass(self, quantum_features: np.ndarray) -> np.ndarray:
        """Classical neural network forward pass"""
        if self.classical_weights is None:
            # Initialize classical weights
            self.classical_weights = np.random.randn(quantum_features.shape[1], 1)

        # Simple linear classifier for now
        output = np.dot(quantum_features, self.classical_weights).flatten()
        return 1 / (1 + np.exp(-output))  # Sigmoid activation

    async def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the quantum neural network"""
        # Encode features
        feature_circuits = await self._encode_features(X)

        # Quantum forward pass
        quantum_features = await self._quantum_forward_pass(feature_circuits)

        # Train classical part (simplified)
        self.classical_weights = np.random.randn(quantum_features.shape[1], 1)

        # Simple gradient descent for classical weights
        learning_rate = 0.01
        epochs = 100

        for epoch in range(epochs):
            predictions = self._classical_forward_pass(quantum_features)
            error = predictions - y
            gradient = np.dot(quantum_features.T, error.reshape(-1, 1)) / len(y)
            self.classical_weights -= learning_rate * gradient

        self.is_fitted = True
        return self

    async def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        feature_circuits = await self._encode_features(X)
        quantum_features = await self._quantum_forward_pass(feature_circuits)
        predictions = self._classical_forward_pass(quantum_features)

        return (predictions > 0.5).astype(int)


class QuantumNAS:
    """Quantum Neural Architecture Search"""

    def __init__(
        self,
        search_space: str = "hybrid_classical_quantum",
        objective: str = "validation_accuracy",
        max_evaluations: int = 50,
    ):
        self.search_space = search_space
        self.objective = objective
        self.max_evaluations = max_evaluations
        self.evaluation_history = []
        self.best_architecture = None
        self.best_score = -np.inf

    def _generate_random_architecture(self) -> ArchitectureCandidate:
        """Generate a random architecture candidate"""
        # Random quantum layers
        n_quantum_layers = np.random.randint(1, 5)
        quantum_layers = []
        for _ in range(n_quantum_layers):
            layer = {
                "type": np.random.choice(["variational", "strongly_entangling"]),
                "n_qubits": np.random.randint(2, 8),
                "n_params": np.random.randint(6, 24),
            }
            quantum_layers.append(layer)

        # Random classical layers
        n_classical_layers = np.random.randint(1, 4)
        classical_layers = []
        for i in range(n_classical_layers):
            layer = {
                "type": "dense",
                "units": np.random.randint(16, 128),
                "activation": np.random.choice(["relu", "tanh", "sigmoid"]),
            }
            classical_layers.append(layer)

        # Random connections (simplified)
        connections = [
            (i, i + 1) for i in range(len(quantum_layers) + len(classical_layers) - 1)
        ]

        return ArchitectureCandidate(
            quantum_layers=quantum_layers,
            classical_layers=classical_layers,
            connections=connections,
        )

    async def _evaluate_architecture(
        self,
        architecture: ArchitectureCandidate,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Evaluate an architecture candidate"""
        try:
            # Create model based on architecture
            n_qubits = (
                architecture.quantum_layers[0]["n_qubits"]
                if architecture.quantum_layers
                else 4
            )
            n_layers = len(architecture.quantum_layers)

            model = QuantumNeuralNetwork(
                n_qubits=n_qubits,
                n_layers=n_layers,
                backend="simulator",  # Use simulator for NAS
            )

            # Train and evaluate
            await model.fit(X_train, y_train)
            predictions = await model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)

            # Calculate complexity penalty
            complexity = self._calculate_complexity(architecture)

            # Combined score (accuracy - complexity penalty)
            score = accuracy - 0.1 * complexity

            return score

        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            return -1.0

    def _calculate_complexity(self, architecture: ArchitectureCandidate) -> float:
        """Calculate architecture complexity score"""
        quantum_complexity = sum(
            layer["n_params"] for layer in architecture.quantum_layers
        )
        classical_complexity = sum(
            layer["units"] for layer in architecture.classical_layers
        )

        return (quantum_complexity + classical_complexity) / 1000.0

    async def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ArchitectureCandidate:
        """Perform neural architecture search"""
        print(f"Starting Quantum NAS with {self.max_evaluations} evaluations...")

        for i in range(self.max_evaluations):
            # Generate candidate architecture
            candidate = self._generate_random_architecture()

            # Evaluate architecture
            score = await self._evaluate_architecture(
                candidate, X_train, y_train, X_val, y_val
            )
            candidate.performance = score
            candidate.complexity_score = self._calculate_complexity(candidate)

            # Update best architecture
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = candidate
                print(f"New best architecture found! Score: {score:.4f}")

            self.evaluation_history.append(candidate)

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{self.max_evaluations} evaluations")

        print(f"Search completed! Best score: {self.best_score:.4f}")
        return self.best_architecture

    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of the search process"""
        scores = [
            arch.performance
            for arch in self.evaluation_history
            if arch.performance is not None
        ]

        return {
            "total_evaluations": len(self.evaluation_history),
            "best_score": self.best_score,
            "mean_score": np.mean(scores) if scores else 0,
            "std_score": np.std(scores) if scores else 0,
            "best_architecture": {
                "n_quantum_layers": (
                    len(self.best_architecture.quantum_layers)
                    if self.best_architecture
                    else 0
                ),
                "n_classical_layers": (
                    len(self.best_architecture.classical_layers)
                    if self.best_architecture
                    else 0
                ),
                "complexity": (
                    self.best_architecture.complexity_score
                    if self.best_architecture
                    else 0
                ),
            },
        }
