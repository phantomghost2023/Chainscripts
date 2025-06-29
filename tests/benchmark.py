from chainscript.testing import QuantumBenchmark

bench = QuantumBenchmark()
results = bench.run(
    test_cases=1000,
    qpu_types=["rigetti", "simulator"],
    compile_modes=["optimized", "unoptimized"]
)
results.to_csv("quantum_benchmarks.csv")