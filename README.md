# EDKA:Efficient Knowledge Distillation and Alignment forImproved KB-VQA: Enhancing Multimodal Understanding

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)](https://pytorch.org/)

EDKA is a state-of-the-art knowledge-augmented language model designed for Visual Question Answering (VQA) tasks. This model combines the power of pre-trained language models with external knowledge to improve the accuracy and reliability of VQA systems.

## Features

- Knowledge-augmented architecture for enhanced VQA performance
- Support for both base and large model variants
- Efficient fact retrieval and integration
- Flexible training pipeline for both reader and retriever components
- Comprehensive evaluation metrics
- Distributed training support

## Project Structure

```
LaKo-main/
├── src/                    # Core source code
│   ├── data/              # Data processing modules
│   ├── model/             # Model architecture definitions
│   ├── evaluation/        # Evaluation metrics and utilities
│   └── util/              # Utility functions
├── data_process/          # Data preprocessing scripts
├── t5_models/             # T5 model implementations
├── script/                # Utility scripts
├── checkpoints/           # Model checkpoints
├── lightning_logs/        # Training logs
└── figure/               # Generated figures and visualizations
```

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd LaKo-main
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Access and Preparation

### Data Access
Training Data is available https://github.com/sparkrickyfly/EKDA/tree/master/data_process

### Data Format
Once you have access, the dataset should be organized as follows:

```
data/
├── train/                 # Training data directory
├── test/                  # Test data directory
└── knowledge_base/        # Knowledge base directory
```

### Data Processing
After obtaining the dataset:

1. Place the data files in the appropriate directories
2. Run the preprocessing script:
```bash
python data_process/deal_vqa.py
```

## Training

### Training the Reader

The reader component can be trained using the provided shell script:

```bash
bash run_okvqa_train.sh
```

Key training parameters:
- `gpu`: GPU device ID
- `model_size`: Model size ("base" or "large")
- `batch_size`: Batch size (automatically set based on model size)
- `n_context`: Number of context items
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `early_stop`: Early stopping patience

### Training the Retriever

To train the retriever component:

```bash
python train_retriever.py
```

### Distributed Training

For distributed training across multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train_reader.py
```

## Evaluation

Evaluate the model's performance:

```bash
python evaluate_retrieved_facts.py
```

The evaluation script provides:
- Exact match accuracy
- F1 score
- Knowledge retrieval metrics
- Response generation quality metrics

## Model Checkpoints

Pre-trained model checkpoints are available in the `checkpoints/` directory:
- Base model: `checkpoint/vqa2.0_base_backbone/`
- Large model: `checkpoint/vqa2.0_large_backbone/`



## Citation

If you use this code in your research, please cite our paper:

[Citation information to be added]


## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Acknowledgments

- Thanks to all contributors who have helped improve this project
- Special thanks to the open-source community for their valuable tools and libraries



