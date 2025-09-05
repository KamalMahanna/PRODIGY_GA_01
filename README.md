# PRODIGY_GA_01: Fine-tuning and Inference with a Language Model

Welcome to PRODIGY_GA_01! This project is designed to guide you through the exciting world of natural language processing, specifically focusing on how to adapt a powerful pre-trained language model to your own specific tasks and then use it to generate new text or make predictions.

At its core, this project provides a hands-on demonstration of two key concepts:

1.  **Fine-tuning**: We take a large, general-purpose language model, specifically a **GPT-2 model**, (which has already learned a vast amount about language from a huge dataset) and further train it on a smaller, task-specific dataset (`train.csv`). This process helps the model specialize and perform exceptionally well on your particular domain or problem. In this project, the fine-tuning is geared towards **Python code generation**, meaning the model will learn to generate Python code based on given instructions. The `finetune.ipynb` notebook walks you through every step, from preparing your data to configuring and training the model.

2.  **Inference**: Once your GPT-2 model is fine-tuned and ready, the `inference.py` script shows you how to put it to work. This involves loading your specialized model and using it to **generate Python code** based on textual instructions you provide. It's where you see the fruits of your fine-tuning efforts in action, transforming natural language prompts into functional code snippets!

This repository is equipped with all the necessary scripts for data preparation, the actual model training, and ultimately, generating insightful predictions. Dive in and explore the power of custom language models for code generation!

## Project Structure

The project is organized as follows:

*   `finetune.ipynb`: A Jupyter notebook containing the code for fine-tuning a pre-trained language model. This notebook guides through data loading, preprocessing, model setup, and training.
*   `inference.py`: A Python script for performing inference using the fine-tuned model. It loads the model and tokenizer, and provides functionality to generate text or make predictions.
*   `requirements.txt`: Lists all the Python dependencies required to run this project.
*   `train.csv`: The dataset used for fine-tuning the model. This file should contain the training data in a suitable format.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `model/`: This directory contains the saved artifacts of the fine-tuned model, including:
    *   `config.json`: Model configuration.
    *   `generation_config.json`: Configuration for text generation.
    *   `merges.txt`: Part of the tokenizer vocabulary (for BPE-based tokenizers).
    *   `model.safetensors`: The model weights in SafeTensors format.
    *   `special_tokens_map.json`: Maps special tokens to their IDs.
    *   `tokenizer_config.json`: Tokenizer configuration.
    *   `tokenizer.json`: The tokenizer vocabulary.
    *   `vocab.json`: Another part of the tokenizer vocabulary.

## Getting Started

Follow these instructions to set up and run the project.

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KamalMahanna/PRODIGY_GA_01.git
    cd PRODIGY_GA_01
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Fine-tuning the Model

To fine-tune the model, open and run the `finetune.ipynb` Jupyter notebook.

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  Open `finetune.ipynb` in your browser.
3.  Run all cells in the notebook to execute the fine-tuning process. The fine-tuned model will be saved in the `model/` directory.

### 2. Performing Inference

After fine-tuning, you can use `inference.py` to perform predictions or generate text with the trained model.

1.  **Run the inference script:**
    ```bash
    python inference.py --input_text "your question"
    ```
    The `inference.py` script will load the model from the `model/` directory and demonstrate how to use it. You may need to modify `inference.py` to suit your specific inference needs (e.g., providing input text, adjusting generation parameters).
