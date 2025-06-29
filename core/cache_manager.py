"""
Predictive Cache Manager for ChainScript
ML-driven caching system that predicts and pre-loads dependencies
"""

import json
import pickle
import time
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import threading
import asyncio
from datetime import datetime, timedelta

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class PredictiveCacheManager:
    """
    Advanced caching system with ML-based prediction and optimization
    """
    
    def __init__(self, cache_dir: str = None, max_cache_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir or Path.home() / ".chainscript" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.cache_index: Dict[str, CacheEntry] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        
        # ML models for prediction
        self.usage_predictor = None
        self.dependency_predictor = None
        self.feature_encoders = {}
        
        # Background optimization
        self.optimization_thread = None
        self.should_optimize = True
        
        self._load_cache_index()
        self._load_ml_models()
        self._start_background_optimization()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache with access pattern tracking"""
        if key not in self.cache_index:
            return None
        
        entry = self.cache_index[key]
        
        # Check if expired
        if entry.is_expired():
            self.invalidate(key)
            return None
        
        # Update access patterns
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        entry.access_count += 1
        entry.last_accessed = current_time
        
        # Load data
        try:
            with open(entry.file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Trigger predictive pre-loading
            self._predict_and_preload(key)
            
            return data
        except Exception as e:
            print(f"Cache read error for {key}: {e}")
            self.invalidate(key)
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600, 
            dependencies: List[str] = None, metadata: Dict = None):
        """Store item in cache with dependency tracking"""
        
        # Serialize data
        file_path = self.cache_dir / f"{key}.pkl"
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Cache write error for {key}: {e}")
            return False
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            file_path=str(file_path),
            size=file_path.stat().st_size,
            created_at=time.time(),
            ttl=ttl_seconds,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.cache_index[key] = entry
        
        # Update dependency graph
        for dep in (dependencies or []):
            self.dependency_graph[dep].append(key)
        
        # Enforce cache size limits
        self._enforce_cache_limits()
        
        # Save updated index
        self._save_cache_index()
        
        return True
    
    def invalidate(self, key: str) -> bool:
        """Remove item from cache and cascade to dependents"""
        if key not in self.cache_index:
            return False
        
        entry = self.cache_index[key]
        
        # Remove file
        try:
            Path(entry.file_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Error removing cache file for {key}: {e}")
        
        # Remove from index
        del self.cache_index[key]
        
        # Cascade invalidation to dependents
        for dependent in self.dependency_graph.get(key, []):
            self.invalidate(dependent)
        
        # Clean up dependency graph
        if key in self.dependency_graph:
            del self.dependency_graph[key]
        
        return True
    
    def _predict_and_preload(self, accessed_key: str):
        """Use ML to predict next likely accessed items and preload them"""
        if not HAS_SKLEARN or not self.usage_predictor:
            return
        
        try:
            # Get features for current context
            features = self._extract_features(accessed_key)
            
            # Predict next likely items
            predictions = self.usage_predictor.predict_proba([features])[0]
            
            # Get top predictions above threshold
            threshold = 0.3
            likely_keys = []
            
            for i, prob in enumerate(predictions):
                if prob > threshold:
                    # Convert back to original key
                    if 'key_encoder' in self.feature_encoders:
                        key = self.feature_encoders['key_encoder'].inverse_transform([i])[0]
                        likely_keys.append((key, prob))
            
            # Sort by probability and preload top items
            likely_keys.sort(key=lambda x: x[1], reverse=True)
            
            for key, prob in likely_keys[:3]:  # Preload top 3
                threading.Thread(
                    target=self._preload_item, 
                    args=(key,), 
                    daemon=True
                ).start()
                
        except Exception as e:
            print(f"Prediction error: {e}")
    
    def _preload_item(self, key: str):
        """Preload an item into memory cache"""
        if key in self.cache_index:
            # Simulate preloading by warming up the cache
            self.get(key)
    
    def _extract_features(self, key: str) -> List[float]:
        """Extract features for ML prediction"""
        features = []
        
        # Time-based features
        current_time = time.time()
        hour = datetime.fromtimestamp(current_time).hour
        day_of_week = datetime.fromtimestamp(current_time).weekday()
        
        features.extend([hour, day_of_week])
        
        # Access pattern features
        if key in self.access_patterns:
            recent_accesses = [t for t in self.access_patterns[key] 
                             if current_time - t < 3600]  # Last hour
            features.extend([
                len(recent_accesses),
                len(self.access_patterns[key]),
                current_time - max(self.access_patterns[key]) if self.access_patterns[key] else 0
            ])
        else:
            features.extend([0, 0, 0])
        
        # Dependency features
        dep_count = len(self.dependency_graph.get(key, []))
        features.append(dep_count)
        
        return features
    
    def train_predictive_models(self):
        """Train ML models on historical access patterns"""
        if not HAS_SKLEARN:
            print("scikit-learn not available for ML features")
            return
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if len(X) < 10:  # Need minimum data
            return
        
        # Train usage predictor
        self.usage_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.usage_predictor.fit(X, y)
        
        # Save models
        self._save_ml_models()
        
        print(f"Trained predictive models on {len(X)} samples")
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[str]]:
        """Prepare training data from access patterns"""
        X, y = [], []
        
        # Create sequences of accesses for training
        for key, timestamps in self.access_patterns.items():
            if len(timestamps) < 2:
                continue
            
            for i in range(len(timestamps) - 1):
                # Features: context when this key was accessed
                features = self._extract_features_for_timestamp(key, timestamps[i])
                X.append(features)
                
                # Label: next key that was accessed
                next_access_time = timestamps[i + 1]
                next_key = self._find_next_accessed_key(next_access_time)
                if next_key:
                    y.append(next_key)
        
        return X, y
    
    def _find_next_accessed_key(self, target_time: float) -> Optional[str]:
        """Find which key was accessed closest to target time"""
        best_key = None
        min_diff = float('inf')
        
        for key, timestamps in self.access_patterns.items():
            for t in timestamps:
                diff = abs(t - target_time)
                if diff < min_diff and diff < 300:  # Within 5 minutes
                    min_diff = diff
                    best_key = key
        
        return best_key
    
    def _extract_features_for_timestamp(self, key: str, timestamp: float) -> List[float]:
        """Extract features for a specific timestamp"""
        # Similar to _extract_features but for historical data
        hour = datetime.fromtimestamp(timestamp).hour
        day_of_week = datetime.fromtimestamp(timestamp).weekday()
        
        return [hour, day_of_week, 0, 0, 0, 0]  # Simplified for demo
    
    def _enforce_cache_limits(self):
        """Remove old items to stay within size limits"""
        total_size = sum(entry.size for entry in self.cache_index.values())
        
        if total_size <= self.max_cache_size:
            return
        
        # Sort by LRU + access frequency
        items = list(self.cache_index.items())
        items.sort(key=lambda x: (x[1].last_accessed, x[1].access_count))
        
        # Remove oldest items
        for key, entry in items:
            self.invalidate(key)
            total_size -= entry.size
            
            if total_size <= self.max_cache_size * 0.8:  # Leave some headroom
                break
    
    def _start_background_optimization(self):
        """Start background thread for cache optimization"""
        def optimize():
            while self.should_optimize:
                time.sleep(300)  # Run every 5 minutes
                try:
                    self._cleanup_expired()
                    if len(self.access_patterns) > 50:  # Have enough data
                        self.train_predictive_models()
                except Exception as e:
                    print(f"Background optimization error: {e}")
        
        self.optimization_thread = threading.Thread(target=optimize, daemon=True)
        self.optimization_thread.start()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        expired_keys = [
            key for key, entry in self.cache_index.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self.invalidate(key)
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump({
                    key: entry.to_dict() 
                    for key, entry in self.cache_index.items()
                }, f)
        except Exception as e:
            print(f"Error saving cache index: {e}")
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if not index_file.exists():
            return
        
        try:
            with open(index_file, 'r') as f:
                data = json.load(f)
            
            for key, entry_data in data.items():
                entry = CacheEntry.from_dict(entry_data)
                if Path(entry.file_path).exists():
                    self.cache_index[key] = entry
        except Exception as e:
            print(f"Error loading cache index: {e}")
    
    def _save_ml_models(self):
        """Save trained ML models"""
        if not self.usage_predictor:
            return
        
        models_file = self.cache_dir / "ml_models.pkl"
        try:
            with open(models_file, 'wb') as f:
                pickle.dump({
                    'usage_predictor': self.usage_predictor,
                    'feature_encoders': self.feature_encoders
                }, f)
        except Exception as e:
            print(f"Error saving ML models: {e}")
    
    def _load_ml_models(self):
        """Load trained ML models"""
        models_file = self.cache_dir / "ml_models.pkl"
        if not models_file.exists():
            return
        
        try:
            with open(models_file, 'rb') as f:
                data = pickle.load(f)
            
            self.usage_predictor = data.get('usage_predictor')
            self.feature_encoders = data.get('feature_encoders', {})
        except Exception as e:
            print(f"Error loading ML models: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_size = sum(entry.size for entry in self.cache_index.values())
        
        return {
            "total_entries": len(self.cache_index),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_hit_patterns": len(self.access_patterns),
            "has_ml_models": self.usage_predictor is not None,
            "dependency_chains": len(self.dependency_graph),
            "average_access_count": sum(e.access_count for e in self.cache_index.values()) / max(len(self.cache_index), 1)
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.should_optimize = False
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=1)


class CacheEntry:
    """Represents a single cache entry with metadata"""
    
    def __init__(self, key: str, file_path: str, size: int, created_at: float,
                 ttl: int, dependencies: List[str], metadata: Dict):
        self.key = key
        self.file_path = file_path
        self.size = size
        self.created_at = created_at
        self.ttl = ttl
        self.dependencies = dependencies
        self.metadata = metadata
        self.last_accessed = created_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > (self.created_at + self.ttl)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'file_path': self.file_path,
            'size': self.size,
            'created_at': self.created_at,
            'ttl': self.ttl,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        """Create from dictionary"""
        entry = cls(
            key=data['key'],
            file_path=data['file_path'],
            size=data['size'],
            created_at=data['created_at'],
            ttl=data['ttl'],
            dependencies=data['dependencies'],
            metadata=data['metadata']
        )
        entry.last_accessed = data.get('last_accessed', data['created_at'])
        entry.access_count = data.get('access_count', 0)
        return entry


# Example usage
if __name__ == "__main__":
    cache = PredictiveCacheManager()
    
    # Test basic caching
    cache.set("test_data", {"numbers": [1, 2, 3, 4, 5]}, ttl_seconds=3600)
    result = cache.get("test_data")
    print(f"Retrieved: {result}")
    
    # Test dependency tracking
    cache.set("processed_data", {"processed": True}, 
              dependencies=["test_data"], ttl_seconds=1800)
    
    # Show stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
