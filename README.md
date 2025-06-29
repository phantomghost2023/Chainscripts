# ChainScript - Next-Gen Script Orchestration System

A revolutionary script orchestration platform that merges hyper-compression, AI-powered usability, and self-improving architecture.

## 🚀 Features

- **Nano-Script Architecture**: Atomic, reusable script components with global deduplication
- **AI-Powered Interface**: Natural language to script translation
- **Bytecode Optimization**: Faster execution through ahead-of-time compilation
- **Self-Improving**: Genetic algorithms for script evolution
- **Predictive Caching**: ML-driven dependency pre-loading
- **Decentralized Sharing**: IPFS-based script marketplace
- **Zero-Second Workflows**: Parallel execution with automatic fallbacks

## 📁 Project Structure

```
chainscript/
├── core/                   # Core engine components
│   ├── nano_engine.py     # Nano-script execution engine
│   ├── bytecode_optimizer.py # Bytecode compilation and optimization
│   ├── cache_manager.py   # Predictive caching system
│   └── genetic_optimizer.py # Genetic algorithm for script evolution
├── ai/                    # AI integration layer
│   ├── nlp_processor.py   # Natural language processing
│   ├── script_generator.py # AI script generation
│   └── debug_assistant.py # Auto-debugging capabilities
├── nano_scripts/          # Atomic script library
│   ├── data/             # Data processing scripts
│   ├── api/              # API interaction scripts
│   └── utils/            # Utility scripts
├── ipfs/                 # Decentralized sharing
│   ├── ipfs_client.py    # IPFS integration
│   └── reputation.py     # Reputation system
├── cli/                  # Command line interface
│   └── chainscript_cli.py
└── examples/             # Example workflows
```

## 🛠 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a workflow
python cli/chainscript_cli.py "Clean sales data and generate report"

# Start interactive mode
python cli/chainscript_cli.py --interactive
```

## 💡 Example Usage

```python
# Natural language workflow
"Fetch latest sales data, clean it, and email summary" 
→ RUN fetch_sales.py | clean_csv.py | summarize.py | email_report.py

# Hybrid language/code
"Get top 10 posts" → RUN fetch_posts.py --limit=10 --source=hackernews
```
