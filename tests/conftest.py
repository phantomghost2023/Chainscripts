import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def golden_scripts():
    return {
        "small": [p for p in Path("golden_scripts/small").glob("*.chain")],
        "large": [p for p in Path("golden_scripts/large").glob("*.chain")]
    }