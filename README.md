# Simple Modular RAG Q&A System

A clean, configurable RAG system for immunization manual Q&A with easy component activation/deactivation.

## 🎯 Key Features

- **Simple Configuration**: One config class, easy to modify
- **Component Toggles**: Enable/disable reranker, evaluation, debug logging
- **Quick Setups**: Pre-configured modes for speed, quality, or debugging
- **Modern Python**: Uses `|` unions and `dict[]`/`list[]` syntax
- **Real-time Changes**: Toggle components at runtime

## 🚀 Quick Start

### Basic Usage
```python
from config import RagConfig
from pipeline import RAGPipeline

# Create config and modify what you need
config = RagConfig()
config.use_reranker = True           # Enable reranking
config.enable_evaluation = False     # Disable evaluation
config.enable_debug = False          # Disable debug logging

# Create pipeline and process question
pipeline = RAGPipeline(config)
result = pipeline.process_question(question, documents, vector_store)
```

### Quick Setups
```python
config = RagConfig()

# Choose one:
config.quick_setup_for_speed()      # Fast answers, no reranking
config.quick_setup_for_quality()    # Best quality, with reranking  
config.quick_setup_for_debug()      # Full debugging and evaluation
```

## ⚙️ Simple Configuration

### Main Components (Enable/Disable)
```python
config = RagConfig()

# Pipeline components
config.use_chapter_filtering = True   # LLM filters relevant chapters first
config.use_reranker = True           # Apply reranking to retrieved docs
config.enable_evaluation = False     # Evaluate generated answers
config.enable_debug = False          # Show detailed debug information

# Logging
config.log_stats = True             # Basic retrieval statistics  
config.log_performance = False      # Detailed timing metrics
config.save_results = False         # Save evaluation results to file

# Retrieval parameters
config.top_k_retrieval = 40         # Documents to retrieve initially
config.top_k_reranked = 15          # Documents after reranking
config.max_context_length = 8000    # Maximum context length
```

### Quick Methods
```python
# Set retrieval sizes
config.set_retrieval_size(30, 15)   # Retrieve 30, rerank to 15

# Toggle components
config.toggle_reranker(True)         # Enable reranker
config.toggle_evaluation(False)      # Disable evaluation  
config.toggle_debug(True)           # Enable debug logging

# Check current status
config.print_status()               # Print current configuration
```

## 📋 Usage Examples

### Example 1: Basic Setup
```python
from config import RagConfig
from pipeline import RAGPipeline

config = RagConfig()
# Default: reranker enabled, evaluation disabled

# Modify as needed:
config.use_reranker = False         # Disable for speed
config.enable_debug = True          # Enable for debugging

pipeline = RAGPipeline(config)
```

### Example 2: Quick Configurations
```python
config = RagConfig()

# For speed (no reranking, minimal logging):
config.quick_setup_for_speed()

# For quality (all quality features enabled):
config.quick_setup_for_quality()

# For debugging (evaluation + debug logging):
config.quick_setup_for_debug()
```

### Example 3: Runtime Changes
```python
config = RagConfig()

# Start with basic setup
print("Initial configuration:")
config.print_status()

# Change at runtime
config.toggle_reranker(False)       # Disable reranker
config.toggle_evaluation(True)      # Enable evaluation
config.set_retrieval_size(25, 12)   # Adjust sizes

print("After changes:")
config.print_status()
```

### Example 4: Custom Models
```python
config = RagConfig()

# Change models
config.llm_model = "gemini-2.0-flash-lite"
config.embedding_model = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"  
config.reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Models will be reloaded automatically
```

## 🏃 Running the System

### Main Script
Edit the configuration section in `main.py` (around line 50):

```python
# =================================================================
# 🎛️  EASY CONFIGURATION - Modify these to enable/disable components
# =================================================================

# Quick setups (uncomment one to use):
# config.quick_setup_for_speed()      # Fast answers, no reranking
# config.quick_setup_for_quality()    # Best quality, with reranking  
# config.quick_setup_for_debug()      # Full debugging and evaluation

# Or manually configure components:
config.use_reranker = True           # Enable/disable reranking
config.enable_evaluation = False     # Enable/disable evaluation
config.enable_debug = False          # Enable/disable debug logging
config.log_performance = True       # Enable/disable performance metrics

# Adjust retrieval parameters:
config.set_retrieval_size(30, 15)   # Retrieve 30, rerank to 15
```

Then run:
```bash
python main.py
```

### Examples
```bash
python examples.py
```

## 🔧 Component Details

### Chapter Filtering (`use_chapter_filtering`)
- Uses LLM to identify relevant chapters before retrieval
- Reduces search space for better performance
- **Enable for**: Better precision, relevant results
- **Disable for**: Speed, simpler pipeline

### Reranker (`use_reranker`)  
- Uses cross-encoder to rerank retrieved documents
- Significantly improves relevance but adds compute time
- **Enable for**: Better answer quality
- **Disable for**: Faster responses

### Evaluation (`enable_evaluation`)
- Compares generated answers with human answers
- Provides detailed scoring and metrics
- **Enable for**: Testing, model comparison
- **Disable for**: Production, speed

### Debug Logging (`enable_debug`)
- Shows detailed retrieval information
- Displays document content and metadata
- **Enable for**: Development, troubleshooting
- **Disable for**: Clean output, production

## 📊 Performance Monitoring

Enable performance tracking:
```python
config.log_performance = True
```

Output example:
```
⚡ Performance Metrics
==========================================
Total Time: 2.45s
├─ Retrieval: 0.65s (26.5%)
├─ Reranking: 0.95s (38.8%)
├─ Generation: 0.85s (34.7%)

Documents: 30 → 15
Context Length: 8,543 chars
```

## 🎯 Configuration Scenarios

### Development/Testing
```python
config.quick_setup_for_debug()
# Enables: evaluation, debug logging, performance metrics
# Good for: testing changes, debugging issues
```

### Production
```python
config.quick_setup_for_quality()
config.enable_debug = False
config.log_performance = False
# Enables: reranker for quality, minimal logging
# Good for: live applications
```

### Fast Demo/Prototype
```python
config.quick_setup_for_speed()
# Disables: reranker, chapter filtering, evaluation
# Good for: quick demos, resource-constrained environments
```

### Custom Research Setup
```python
config = RagConfig()
config.use_reranker = True
config.enable_evaluation = True
config.enable_debug = True
config.log_performance = True
config.save_results = True
config.set_retrieval_size(50, 20)  # Larger retrieval
# Good for: comparing models, research
```

## 📁 File Structure

```
├── config.py          # Simple configuration class
├── pipeline.py        # Modular pipeline components  
├── main.py            # Main execution (edit config section)
├── examples.py        # Simple usage examples
├── vector_stores.py   # Document processing
├── templates/
│   ├── prompts.py     # Prompt templates
│   └── titles.py      # Document titles
└── README.md          # This file
```

## 🔄 Migration from Complex Version

The new system is much simpler:

**Old (complex):**
```python
features = FeatureFlags(use_reranker=True, enable_evaluation=False, ...)
config = RagConfig(feature_flags=features)
```

**New (simple):**
```python
config = RagConfig()
config.use_reranker = True
config.enable_evaluation = False
```

**Old preset configs:**
```python
config = create_preset_configs()['development']
```

**New quick setups:**
```python
config = RagConfig()
config.quick_setup_for_debug()  # Similar to development mode
```

## 🧩 Extending the System

To add new components:

1. **Add configuration option** to `RagConfig.__init__()`:
   ```python
   self.use_new_feature = False
   ```

2. **Add toggle method** (optional):
   ```python
   def toggle_new_feature(self, enabled: bool = None):
       # Similar to other toggle methods
   ```

3. **Use in pipeline modules** - check the config flag:
   ```python
   if self.config.use_new_feature:
       # Your new feature logic
   ```

4. **Add to quick setups** as needed.

## 💡 Tips

- **Start simple**: Use default config, modify only what you need
- **Use quick setups**: `quick_setup_for_*()` methods for common scenarios  
- **Toggle at runtime**: Change configuration without restarting
- **Monitor performance**: Enable `log_performance` to see bottlenecks
- **Print status**: Use `config.print_status()` to see current settings

That's it! The system is now much simpler while maintaining all the modular functionality.
