# Simple Modular RAG Q&A System

A clean, configurable RAG system for immunization manual Q&A with easy component activation/deactivation.

## 🎯 Key Features

- **Simple Configuration**: One config class, easy to modify
- **Two Setup Options**: Full features or basic/fast mode
- **Component Toggles**: Enable/disable reranker, evaluation, debug logging
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

### Two Simple Setup Options
```python
config = RagConfig()

# Option 1: Enable all features
config.enable_all()         # Everything: reranker, evaluation, debug, etc.

# Option 2: Basic setup for speed  
config.basic_setup()        # Minimal: no reranker, no evaluation, faster
```

## ⚙️ Simple Configuration

### Default Configuration
When you create `RagConfig()`, you get:
```python
config = RagConfig()
# ✅ Chapter filtering: enabled
# ✅ Reranker: enabled  
# ❌ Evaluation: disabled
# ❌ Debug: disabled
# Retrieval: 40 docs → 15 reranked
```

### Two Setup Methods

#### `config.enable_all()` - Full Features
```python
config = RagConfig()
config.enable_all()
# ✅ All features enabled
# ✅ Evaluation and debug logging
# ✅ Save results to file
# Retrieval: 40 docs → 15 reranked
```

#### `config.basic_setup()` - Fast/Minimal
```python
config = RagConfig()
config.basic_setup()
# ❌ No chapter filtering
# ❌ No reranker
# ❌ No evaluation or debug
# Retrieval: 20 docs (no reranking)
```

### Manual Configuration
```python
config = RagConfig()

# Pick exactly what you need:
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

### Example 1: Default Setup
```python
from config import RagConfig
from pipeline import RAGPipeline

config = RagConfig()
# Default: reranker enabled, evaluation disabled

pipeline = RAGPipeline(config)
```

### Example 2: Full Features
```python
config = RagConfig()
config.enable_all()  # Enable everything

pipeline = RAGPipeline(config)
```

### Example 3: Basic/Fast
```python
config = RagConfig()
config.basic_setup()  # Minimal for speed

pipeline = RAGPipeline(config)
```

### Example 4: Custom Configuration
```python
config = RagConfig()

# Pick exactly what you want
config.use_reranker = False         # Disable for speed
config.enable_evaluation = True     # But enable evaluation
config.enable_debug = True         # And debugging

pipeline = RAGPipeline(config)
```

### Example 5: Runtime Changes
```python
config = RagConfig()

# Start with defaults
config.print_status()

# Change at runtime
config.toggle_reranker(False)       # Disable reranker
config.toggle_evaluation(True)      # Enable evaluation
config.set_retrieval_size(25, 12)   # Adjust sizes

config.print_status()
```

## 🏃 Running the System

### Main Script
Edit the configuration section in `main.py` (around line 50):

```python
# =================================================================
# 🎛️  EASY CONFIGURATION - Choose your setup
# =================================================================

# Choose one setup (uncomment the one you want):
# config.enable_all()         # Enable all features (reranker, evaluation, debug, etc.)
# config.basic_setup()        # Basic setup (no reranker, no evaluation, faster)

# Or manually configure what you need:
config.use_reranker = True           # Enable/disable reranking
config.enable_evaluation = False     # Enable/disable evaluation
config.enable_debug = False          # Enable/disable debug logging
config.log_performance = True       # Enable/disable performance metrics
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
- **Enable for**: Better precision, relevant results
- **Disable for**: Speed, simpler pipeline

### Reranker (`use_reranker`)  
- Uses cross-encoder to rerank retrieved documents
- **Enable for**: Better answer quality
- **Disable for**: Faster responses

### Evaluation (`enable_evaluation`)
- Compares generated answers with human answers
- **Enable for**: Testing, model comparison
- **Disable for**: Production, speed

### Debug Logging (`enable_debug`)
- Shows detailed retrieval information
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
config = RagConfig()
config.enable_all()
# Enables: everything for full debugging and evaluation
```

### Production
```python
config = RagConfig()
# Default setup: reranker enabled, no evaluation/debug
# Good balance of quality and performance
```

### Fast Demo/Prototype
```python
config = RagConfig()
config.basic_setup()
# Minimal components for maximum speed
```

### Custom Research
```python
config = RagConfig()
config.use_reranker = True
config.enable_evaluation = True
config.enable_debug = True
config.log_performance = True
config.save_results = True
config.set_retrieval_size(50, 20)  # Larger retrieval
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

## 📊 Setup Comparison

| Component | Default | enable_all() | basic_setup() |
|-----------|---------|--------------|---------------|
| Chapter Filtering | ✅ | ✅ | ❌ |
| Reranker | ✅ | ✅ | ❌ |
| Evaluation | ❌ | ✅ | ❌ |
| Debug Logging | ❌ | ✅ | ❌ |
| Performance Log | ❌ | ✅ | ❌ |
| Save Results | ❌ | ✅ | ❌ |
| **Retrieval Size** | 40 → 15 | 40 → 15 | 20 |

## 🧩 Extending the System

To add new components:

1. **Add configuration option** to `RagConfig.__init__()`:
   ```python
   self.use_new_feature = False
   ```

2. **Add to setup methods**:
   ```python
   def enable_all(self):
       # ... existing code ...
       self.use_new_feature = True
   
   def basic_setup(self):
       # ... existing code ...
       self.use_new_feature = False
   ```

3. **Use in pipeline modules**:
   ```python
   if self.config.use_new_feature:
       # Your new feature logic
   ```

## 💡 Tips

- **Start simple**: Use default config, modify only what you need
- **Two main options**: `enable_all()` for full features, `basic_setup()` for speed
- **Toggle at runtime**: Change configuration without restarting
- **Monitor performance**: Enable `log_performance` to see bottlenecks
- **Print status**: Use `config.print_status()` to see current settings

That's it! Clean, simple, and powerful. 🚀
