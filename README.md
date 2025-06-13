# Fine-tuning PaliGemma for Image-to-JSON Extraction with QLoRA

This project demonstrates how to fine-tune Google's PaliGemma (a powerful multimodal vision-language model) to extract structured JSON data from images. We utilize QLoRA (Quantized Low-Rank Adaptation) and Supervised Fine-Tuning (SFT) techniques to efficiently adapt the model for this specific task.

## Key Technical Approaches

- **QLoRA (Quantized Low-Rank Adaptation)**: Enables memory-efficient fine-tuning of large language models by quantizing the pre-trained weights and only training a small set of adapter parameters
- **Supervised Fine-Tuning (SFT)**: Training with explicit instruction following to teach the model to extract structured data
- **PaliGemma Model**: Leveraging Google's 3B parameter multimodal model that can process both images and text

## Dataset

The project uses the CORD dataset, which contains receipt images paired with structured JSON data representing the receipt content, making it ideal for training a model to extract structured information from visual inputs.

## Implementation Details

The implementation includes:
- Model quantization with bitsandbytes
- Parameter-efficient fine-tuning with PEFT
- Training pipeline with TRL's SFTTrainer
- Evaluation on validation data

# Getting Started

For the complete implementation and step-by-step guide, please refer to the [peft_paligemma_im2json_qlora_SFT.ipynb](peft_paligemma_im2json_qlora_SFT.ipynb) notebook in this directory.
