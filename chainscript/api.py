"""
ChainScript API: Quantum Microservices
=====================================

APIs for deploying quantum computation capabilities as microservices with
FastAPI. Supports seamless integration with major QPU providers.
"""

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import uvicorn
from pyquil import Program
from pyquil.api import QuantumComputer
from .core import QuantumEngine

app = FastAPI()

engine = QuantumEngine()


@app.get("/status")
async def status() -> Dict[str, Any]:
    """Get server and backend status"""
    return {"server": "running", "backend_status": engine.get_backend_status()}


@app.post("/execute")
async def execute_program(program: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a quantum program sent as JSON"""
    try:
        quantum_program = Program(program["instructions"])
        result = await engine.run_circuit(quantum_program)
        return {
            "result": result.result.tolist(),
            "execution_time": result.execution_time,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error executing program: {e}")


@app.get("/available_qpus")
async def available_qpus() -> List[str]:
    """List available QPU backends"""
    return [backend.value for backend in engine.processor.backends.keys()]


# Serve the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
