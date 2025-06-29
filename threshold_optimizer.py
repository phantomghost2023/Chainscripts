#!/usr/bin/env python3
"""
ChainScript Threshold Optimizer
================================
Advanced tool to analyze and optimize classification thresholds for accurate script profiling.
"""

import json
import ast
import os
from typing import Dict, List, Tuple, Any
from core.nano_engine import NanoScriptEngine
import numpy as np

class ThresholdOptimizer:
    def __init__(self):
        self.engine = NanoScriptEngine()
        self.test_scripts = []
        self.current_thresholds = {
            'small': {'lines': 50, 'complexity': 10, 'ast_nodes': 100},
            'medium-lightweight': {'lines': 150, 'complexity': 25, 'ast_nodes': 300},
            'medium-balanced': {'lines': 300, 'complexity': 50, 'ast_nodes': 600},
            'large': {'lines': 500, 'complexity': 100, 'ast_nodes': 1000}
        }
        
    def load_test_cases(self):
        """Load test scripts with expected classifications"""
        # Recursively find all Python scripts in nano_scripts directory
        script_files = []
        nano_scripts_dir = 'nano_scripts'
        
        if os.path.exists(nano_scripts_dir):
            for root, dirs, files in os.walk(nano_scripts_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        script_files.append(os.path.join(root, file))
        
        print(f"Found {len(script_files)} Python scripts in nano_scripts directory:")
        for script in script_files:
            print(f"  - {script}")
        
        # Classify scripts based on analysis (dynamic classification)
        for script_file in script_files:
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                metrics = self._analyze_script_metrics(content)
                expected_class = self.classify_with_current_thresholds(metrics)
                
                case = {
                    'file': script_file,
                    'expected': expected_class,
                    'description': f'Auto-classified from {os.path.basename(script_file)}',
                    'content': content,
                    'metrics': metrics
                }
                
                self.test_scripts.append(case)
                print(f"  📄 {script_file} -> {expected_class}")
                
            except Exception as e:
                print(f"  ❌ Error loading {script_file}: {e}")
                continue
                
    def _analyze_script_metrics(self, code: str) -> Dict[str, Any]:
        """Analyze comprehensive script metrics"""
        try:
            tree = ast.parse(code)
            
            # Basic metrics
            lines = len([line for line in code.split('\n') if line.strip()])
            
            # AST analysis
            node_counts = {}
            complexity_score = 0
            
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
                
                # Complexity scoring
                if isinstance(node, (ast.For, ast.While, ast.If)):
                    complexity_score += 1
                elif isinstance(node, (ast.Try, ast.With)):
                    complexity_score += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity_score += 2
                elif isinstance(node, ast.ClassDef):
                    complexity_score += 3
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                    complexity_score += 1
                    
            total_nodes = sum(node_counts.values())
            
            # Pattern analysis
            patterns = {
                'has_classes': node_counts.get('ClassDef', 0) > 0,
                'has_functions': node_counts.get('FunctionDef', 0) > 2,
                'has_loops': (node_counts.get('For', 0) + node_counts.get('While', 0)) > 0,
                'has_comprehensions': any(node_counts.get(t, 0) > 0 for t in ['ListComp', 'DictComp', 'SetComp']),
                'has_imports': node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0) > 3,
                'has_decorators': any('decorator' in str(node).lower() for node in ast.walk(tree))
            }
            
            return {
                'lines': lines,
                'complexity_score': complexity_score,
                'total_ast_nodes': total_nodes,
                'node_counts': node_counts,
                'patterns': patterns,
                'function_count': node_counts.get('FunctionDef', 0),
                'class_count': node_counts.get('ClassDef', 0),
                'import_count': node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0)
            }
            
        except Exception as e:
            return {
                'lines': len([line for line in code.split('\n') if line.strip()]),
                'complexity_score': 1,
                'total_ast_nodes': 10,
                'error': str(e)
            }
    
    def classify_with_current_thresholds(self, metrics: Dict[str, Any]) -> str:
        """Classify script using current threshold settings"""
        lines = metrics['lines']
        complexity = metrics['complexity_score']
        nodes = metrics['total_ast_nodes']
        
        if (lines <= self.current_thresholds['small']['lines'] and 
            complexity <= self.current_thresholds['small']['complexity'] and
            nodes <= self.current_thresholds['small']['ast_nodes']):
            return 'small'
        elif (lines <= self.current_thresholds['medium-lightweight']['lines'] and
              complexity <= self.current_thresholds['medium-lightweight']['complexity'] and
              nodes <= self.current_thresholds['medium-lightweight']['ast_nodes']):
            return 'medium-lightweight'
        elif (lines <= self.current_thresholds['medium-balanced']['lines'] and
              complexity <= self.current_thresholds['medium-balanced']['complexity'] and
              nodes <= self.current_thresholds['medium-balanced']['ast_nodes']):
            return 'medium-balanced'
        else:
            return 'large'
    
    def analyze_misclassifications(self):
        """Analyze current classification accuracy"""
        print("🔍 ANALYZING CURRENT CLASSIFICATION ACCURACY")
        print("=" * 60)
        
        correct = 0
        total = len(self.test_scripts)
        
        for script in self.test_scripts:
            actual = self.classify_with_current_thresholds(script['metrics'])
            expected = script['expected']
            is_correct = actual == expected
            
            if is_correct:
                correct += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
                
            print(f"\n📄 {os.path.basename(script['file'])}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")
            print(f"   Status:   {status}")
            print(f"   Lines:    {script['metrics']['lines']}")
            print(f"   Complexity: {script['metrics']['complexity_score']}")
            print(f"   AST Nodes: {script['metrics']['total_ast_nodes']}")
            
        accuracy = (correct / total) * 100
        print(f"\n📊 ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
        return accuracy
    
    def suggest_optimal_thresholds(self):
        """Use data analysis to suggest better thresholds"""
        print("\n🎯 CALCULATING OPTIMAL THRESHOLDS")
        print("=" * 60)
        
        # Group scripts by expected classification
        categories = {}
        for script in self.test_scripts:
            cat = script['expected']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(script['metrics'])
        
        # Calculate statistics for each category
        optimal_thresholds = {}
        
        for category, metrics_list in categories.items():
            if not metrics_list:
                continue
                
            lines_values = [m['lines'] for m in metrics_list]
            complexity_values = [m['complexity_score'] for m in metrics_list]
            nodes_values = [m['total_ast_nodes'] for m in metrics_list]
            
            # Use 95th percentile as upper bound for each category
            optimal_thresholds[category] = {
                'lines': int(np.percentile(lines_values, 95) * 1.2),  # 20% buffer
                'complexity': int(np.percentile(complexity_values, 95) * 1.2),
                'ast_nodes': int(np.percentile(nodes_values, 95) * 1.2)
            }
            
            print(f"\n📈 {category.upper()} CATEGORY:")
            print(f"   Lines range: {min(lines_values)}-{max(lines_values)} (suggested max: {optimal_thresholds[category]['lines']})")
            print(f"   Complexity range: {min(complexity_values)}-{max(complexity_values)} (suggested max: {optimal_thresholds[category]['complexity']})")
            print(f"   AST nodes range: {min(nodes_values)}-{max(nodes_values)} (suggested max: {optimal_thresholds[category]['ast_nodes']})")
        
        return optimal_thresholds
    
    def test_new_thresholds(self, new_thresholds: Dict):
        """Test accuracy with proposed new thresholds"""
        old_thresholds = self.current_thresholds.copy()
        self.current_thresholds = new_thresholds
        
        print("\n🧪 TESTING NEW THRESHOLDS")
        print("=" * 60)
        
        accuracy = self.analyze_misclassifications()
        
        # Restore original thresholds
        self.current_thresholds = old_thresholds
        
        return accuracy
    
    def generate_enhanced_nano_engine_config(self, optimal_thresholds: Dict):
        """Generate updated configuration for nano_engine.py"""
        config = {
            "classification_thresholds": {
                "small_script": {
                    "max_lines": optimal_thresholds.get('small', {}).get('lines', 60),
                    "max_complexity": optimal_thresholds.get('small', {}).get('complexity', 12),
                    "max_ast_nodes": optimal_thresholds.get('small', {}).get('ast_nodes', 120)
                },
                "medium_lightweight": {
                    "max_lines": optimal_thresholds.get('medium_lightweight', {}).get('lines', 180),
                    "max_complexity": optimal_thresholds.get('medium_lightweight', {}).get('complexity', 30),
                    "max_ast_nodes": optimal_thresholds.get('medium_lightweight', {}).get('ast_nodes', 360)
                },
                "medium_balanced": {
                    "max_lines": optimal_thresholds.get('medium_balanced', {}).get('lines', 360),
                    "max_complexity": optimal_thresholds.get('medium_balanced', {}).get('complexity', 60),
                    "max_ast_nodes": optimal_thresholds.get('medium_balanced', {}).get('ast_nodes', 720)
                },
                "large_script": {
                    "max_lines": 1000,  # Large threshold is open-ended
                    "max_complexity": 200,
                    "max_ast_nodes": 2000
                }
            },
            "optimization_strategies": {
                "small": "fast-jit",
                "medium_lightweight": "lightweight", 
                "medium_balanced": "balanced",
                "large": "full-optimization"
            }
        }
        
        print("\n📝 GENERATED OPTIMAL CONFIGURATION:")
        print("=" * 60)
        print(json.dumps(config, indent=2))
        
        return config

def main():
    print("🎯 ChainScript Threshold Optimizer")
    print("=" * 60)
    print("Analyzing script classification accuracy and optimizing thresholds...\n")
    
    optimizer = ThresholdOptimizer()
    
    # Load test cases
    print("📂 Loading test scripts...")
    optimizer.load_test_cases()
    print(f"   Loaded {len(optimizer.test_scripts)} test scripts")
    
    # Analyze current accuracy
    current_accuracy = optimizer.analyze_misclassifications()
    
    # Calculate optimal thresholds
    optimal_thresholds = optimizer.suggest_optimal_thresholds()
    
    # Test new thresholds
    new_accuracy = optimizer.test_new_thresholds(optimal_thresholds)
    
    # Generate configuration
    config = optimizer.generate_enhanced_nano_engine_config(optimal_thresholds)
    
    # Summary
    print(f"\n🎯 OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Current Accuracy: {current_accuracy:.1f}%")
    print(f"Projected Accuracy: {new_accuracy:.1f}%")
    improvement = new_accuracy - current_accuracy
    print(f"Improvement: {improvement:+.1f} percentage points")
    
    if improvement > 0:
        print("\n✅ RECOMMENDED: Apply optimized thresholds")
        # Save configuration to file
        with open('optimized_thresholds.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("💾 Saved optimized configuration to 'optimized_thresholds.json'")
    else:
        print("\n⚠️  Current thresholds appear optimal")
    
    print("\n🚀 Threshold optimization complete!")

if __name__ == "__main__":
    main()
