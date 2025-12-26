# Recommender-Systems-with-Large-Language-Models
======================================================================
A hybrid recommendation system combining LightGCN (Graph Neural Network) 
and Large Language Models (LLMs) (Qwen3-14B) for movie recommendation.
Supports cold-start scenarios, semantic preference understanding, and multi-dimensional evaluation.
======================================================================

## 1. Project Overview
----------------------------------------------------------------------
Core capabilities:
- Capture user-item interaction patterns via LightGCN (graph-based recommendation)
- Understand natural language preferences (e.g., "sci-fi movies with time travel") via fine-tuned Qwen3-14B
- Generate personalized, explainable movie recommendations
- Comprehensive evaluation (Recall/HR/NDCG/coverage/diversity)
- Support cold-start for new movies via LLM semantic encoding

## 2. Project Structure
----------------------------------------------------------------------
Recommender-Systems-with-Large-Language-Models/
├── data/                          # Data directory (sample + instructions)
│   └── README.txt                 # Data preparation guide
├── src/                           # Auxiliary resources
│   ├── temperature_gradient.png   # Hyperparameter tuning visualization
│   └── trainer_log.jsonl          # LLM fine-tuning logs
├── bailian_train.jsonl            # LLM fine-tuning dataset (instruction format)
├── comprehensive_evaluation.py    # Full evaluation (all metrics + visualization)
├── data_loader.py                 # Data loading + graph construction (LightGCN)
├── demo.py                        # End-to-end recommendation demo
├── diagnostic_visualization.py    # Evaluation result plotting
├── eval_english_only.py           # English-only evaluation script
├── generate_recommendations.py    # Recommendation generation (LLM + LightGCN)
├── model.py                       # LightGCN model definition
├── requirements_simple.txt        # Dependencies list
├── semantic_encoder.py            # LLM-based semantic encoding (Qwen3)
├── simple_eval.py                 # Quick evaluation (core metrics only)
├── train_lightgcn.py              # LightGCN training script
└── .gitignore                     # Git ignore configuration (large files, cache)

## 3. Environment Setup
----------------------------------------------------------------------
### Prerequisites
- Python 3.8+
- CUDA 11.3+ (GPU acceleration required)
- Git (for repository cloning)

### Install Dependencies
Step 1: Clone the repository
$ git clone https://github.com/DemonL1/Recommender-Systems-with-Large-Language-Models.git
$ cd Recommender-Systems-with-Large-Language-Models

Step 2: Install required packages
$ pip install -r requirements_simple.txt

Key Dependencies (auto-installed via requirements):
- PyTorch 1.12.1+ & PyTorch Geometric (for LightGCN)
- Hugging Face Transformers (for Qwen3-14B)
- Scikit-learn, NumPy (metrics calculation)
- Matplotlib, Seaborn (visualization)
- FastAPI (optional, for API deployment)
- Tqdm (progress tracking)

## 4. Quick Start Guide
----------------------------------------------------------------------
### 4.1 Train LightGCN Model
$ python train_lightgcn.py
- Hyperparameters (batch size, lr, epochs) can be adjusted in the script
- Trained checkpoints saved to ./checkpoints/
- Automatically generates graph data from data/ directory

### 4.2 Fine-tune Qwen3-14B (Optional)
Use LLaMA-Factory for instruction tuning (compatible with project dataset):
$ python -m src.train \
  --stage sft \
  --model_name_or_path /path/to/Qwen3-14B \
  --dataset fine_tune_dataset_instruction \
  --finetuning_type lora \
  --quantization_bit 4

### 4.3 Generate Recommendations
$ python generate_recommendations.py
- Input user preferences via command line or modify script
- Example input: "I like heartwarming animated movies from Studio Ghibli"
- Output: Top-10 personalized recommendations + explanations

### 4.4 Run Demo
$ python demo.py
- Loads pre-trained LightGCN and fine-tuned Qwen3-14B
- Interactive interface for preference input
- Displays user history + recommendations + scores

### 4.5 Evaluate Model
Option 1: Comprehensive evaluation (all metrics + plots)
$ python comprehensive_evaluation.py
- Outputs Recall@K, Precision@K, HR@K, NDCG@K, coverage, diversity
- Generates visualization plots in ./evaluation_plots/
- Saves detailed report (evaluation_report.txt/json)

Option 2: Quick evaluation (core metrics)
$ python simple_eval.py
- Faster execution (no visualization)
- Outputs key metrics (Recall/HR/NDCG) for specified K values

## 5. API Deployment (Optional)
----------------------------------------------------------------------
Deploy as a FastAPI service for external calls:
1. Create a file named recommend_api.py (use project's API template)
2. Run the service:
$ uvicorn recommend_api:app --host 0.0.0.0 --port 8000

API Endpoint:
- POST /recommend
  Request Body: {"user_input": "your preference", "max_length": 500, "temperature": 0.6}
  Response: {"status": "success", "recommendation": "top-10 movies + explanations"}

## 6. Evaluation Metrics
----------------------------------------------------------------------
| Metric       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| Recall@K     | Ratio of relevant items (user-liked) in top-K recommendations               |
| Precision@K  | Ratio of relevant items among top-K recommendations                          |
| HR@K         | Hit Rate: Whether any relevant item is in top-K (1=hit, 0=miss)             |
| NDCG@K       | Normalized Discounted Cumulative Gain: Ranks relevance (higher=better order) |
| Coverage     | Percentage of total movies recommended across all users (diversity of items) |
| Diversity    | 1 - average similarity of recommended items (higher=more diverse)            |

## 7. Key Notes
----------------------------------------------------------------------
### Data Preparation
- Follow data/README.txt to prepare user-item interaction data (ratings.csv)
- For LLM fine-tuning: Prepare instruction-response pairs (like bailian_train.jsonl)
- Do NOT upload large files (model checkpoints, raw datasets >100MB) to GitHub
- Use .gitignore to exclude __pycache__, .pth, .pt, and data/*.csv (large files)

### Hardware Requirements
- LightGCN Training: 8GB+ GPU memory
- Qwen3-14B Fine-tuning: 24GB+ GPU memory (4-bit quantization recommended)
- Inference: 12GB+ GPU memory (for Qwen3-14B + LightGCN)

### LLM Fine-tuning Tips
- Use LoRA (Low-Rank Adaptation) to save GPU memory
- Adjust temperature (0.6-0.8) for recommendation diversity
- Fine-tune on movie preference data for better semantic understanding

## 8. Troubleshooting
----------------------------------------------------------------------
- CUDA Out of Memory: Reduce batch size, enable gradient checkpointing, or use 4-bit quantization
- LLM Loading Error: Ensure Qwen3-14B model files are complete; add trust_remote_code=True
- Evaluation Metrics NaN: Check data format (no missing user/item IDs, valid ratings)
- Graph Construction Failed: Verify data_loader.py has correct file paths for data/ directory
- Dependency Conflicts: Use virtual environment (conda create -n recsys python=3.8)

## 9. Future Improvements
----------------------------------------------------------------------
- Integrate more LLMs (Llama 3, Mistral) for cross-model comparison
- Add real-time user feedback collection to update recommendations
- Optimize inference speed for production deployment
- Support multi-modal recommendations (text + movie posters)
- Add user preference embedding visualization

## 10. License
----------------------------------------------------------------------
This project is for research and educational purposes only.
Please comply with the license agreements of:
- Qwen3-14B (Alibaba Cloud)
- LightGCN (original paper license)
- Hugging Face Transformers, PyTorch, and other open-source libraries