## Scaled Bayesian Planner for Multimodal Theory-of-Mind

This repository contains the code and data for the paper. The project focuses on developing scalable Bayesian reasoning models, using post-trained language models as policy models to evaluate multimodal Theory-of-Mind (ToM) tasks, including those in unseen scenarios.

### Table of Contents
- [Scaling Multimodal Theory-of-Mind with Weak-to-Strong Bayesian Reasoning](#scaling-multimodal-theory-of-mind-with-weak-to-strong-bayesian-reasoning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Data](#data)
  - [Post-training Language Models into Policy Models](#post-training-language-models-into-policy-models)
  - [Evaluation](#evaluation)
  - [Getting Started](#getting-started)

---

### Overview

This repository contains:
- **Processed Data** for training and evaluation of multimodal Theory-of-Mind models, including data for unseen scenarios.
- **Post-training Instructions** for converting language models (LMs) into policy models suitable for Bayesian reasoning.
- **Evaluation Instructions** for testing scaled Bayesian reasoning models on Theory-of-Mind tasks across multiple scenarios, including previously unseen ones.

---

### Data

The processed data used for training and evaluation is organized in the `data/` folder as follows:

```
data/
├── eval
│   ├── multimodal_representations_virtual_home.json
│   └── unseen
│       ├── ancient_egyptian.json
│       ├── andersen_fairy_tales.json
│       ├── medieval_castle.json
│       ├── outer_space.json
│       └── wild_west.json
└── post_train
    ├── ancient_egyptian_finetuning_data.json
    ├── andersen_fairy_tales_finetuning_data.json
    ├── formatted_finetuning_data.json
    ├── medieval_castle_finetuning_data.json
    ├── outer_space_finetuning_data.json
    ├── virtual_home_finetuing_data_MMToM.json
    └── wild_west_finetuning_data.json
```

- **Training data**: Located under `data/post_train/`.
- **Evaluation data**: Located under `data/eval/`, with a specific subset for unseen scenarios in the `unseen/` folder.

For details on how the data was created from multimodal simulators, please refer to the upcoming updates.

---

### Post-training Language Models into Policy Models

To post-train the language models (LLMs) into policy models for Bayesian reasoning, use the following command:

```bash
python post_train_policy_model.py \
  --train_file post_train_set.json \
  --model_name_or_path llama3.1/Meta-Llama-3.1-8B \
  --fisher_matrix_path fisher-matrix/fisher-matrix-6B \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 5e-5 \
  --num_epochs 3
```

Key parameters:
- `--train_file`: Path to the post-training dataset.
- `--model_name_or_path`: Pretrained language model (e.g., Meta-Llama 3.1 8B).
- `--fisher_matrix_path`: Path to the Fisher matrix used for post-training optimization.
- `--per_device_train_batch_size`: Batch size per device for training.
- `--gradient_accumulation_steps`: Steps to accumulate gradients for larger effective batch sizes.
- `--lr`: Learning rate for fine-tuning.
- `--num_epochs`: Number of training epochs.

---

### Evaluation

To evaluate the scaled Bayesian reasoning models on multimodal Theory-of-Mind tasks, including unseen scenarios, use the following command:

```bash
python scale_bayesian_policy_model.py \
  --eval_data_path multimodal_representations_virtual_home.json \
  --model_name_or_path llama3.1/Meta-Llama-3.1-8B \
  --large_base_name_or_path llama3.1/Meta-Llama-3.1-405B
```

Key parameters:
- `--eval_data_path`: Path to the evaluation data.
- `--model_name_or_path`: Pretrained language model for evaluation.
- `--large_base_name_or_path`: Larger LLM used for scaling Bayesian reasoning.

---

### Getting Started

1. **Install dependencies**:
   Ensure you have the required Python dependencies. You can set up a virtual environment and install them with:
   ```bash
    conda env create -f scale_lm_policy_model/envs.yaml
   ```

2. **Download the data**:
   Download and extract the data into the `data/` folder:
   ```bash
   unzip scale_tom_data.zip -d data/
   ```

3. **Post-train the language model**:
   Use the instructions provided in the [Post-training](#post-training-language-models-into-policy-models) section to convert LLMs into policy models.

4. **Run evaluations**:
   Follow the [Evaluation](#evaluation) instructions to test the Bayesian models.

---
