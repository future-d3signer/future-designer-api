# Experiments

This directory contains various experiments and research code for the Future Designer API project.

## Structure

### Evaluation Scripts
- **`evaluation_clip_all.py`** - CLIP model evaluation across all caption attributes of furniture
- **`evaluation_clip_single.py`** - CLIP model evaluation for single caption attribute of furniture

### Model Training
- **`paligemma_lora.py`** - PaliGemma model fine-tuning with LoRA
- **`qwen_lora.py`** - Qwen model fine-tuning with LoRA

### Data Generation
- **`synthetic_data.py`** - Synthetic furniture data generation
- **`furniture_attributes.json`** - Furniture attribute definitions and schemas

### Notebooks (`notebooks/`)
- **`furniture-label.ipynb`** - Furniture labeling experiments
- **`image_to_text.ipynb`** - Image-to-text generation experiments  
- **`qwen_lora.ipynb`** - Qwen LoRA training notebook
- **`sam2.ipynb`** - SAM2 segmentation experiments

### VLLM Labelling (`vllm_labelling/`)
Vision-Language Model labeling experiments:
- **`intern.py`** - InternVL model experiments
- **`llava.py`** - LLaVA model experiments
- **`molmo.py`** - Molmo model experiments
- **`phi.py`** - Phi-Vision model experiments

### Dataset

Future Designer synthetic dataset can be found at: [**furniture-synthetic-dataset**](https://huggingface.co/datasets/filnow/furniture-synthetic-dataset)

**Furniture Synthetic Dataset**

A curated dataset of furniture images in four categories, with detailed attribute annotations.

Designed for training and evaluating small Vision Language Models (VLMs) in extracting structured information from furniture images.

**Dataset Description**

*Overview*
- Total Images: 10,000
- Training Set: 9,000 images (generated)
- Test Set: 1,000 images (real photographs)
- Image Generation: Stable Diffusion Medium 3.5
- Test Set Annotation: Qwen2 VL 72B

*Features*

Each image is size 448x448 and annotated with the following attributes:
- `type`: Furniture category (bed, table, sofa, chair)
- `style`: Design style of the furniture
- `color`: Predominant colors
- `material`: Main materials used
- `shape`: Physical form characteristics
- `details`: Additional decorative or functional elements
- `room_type`: Intended room placement
- `price_range`: Estimated price category
- `prompt`: Original generation prompt (for synthetic images)

*Intended Uses*

This dataset is particularly useful for:
- Fine-tuning small VLMs for structured furniture attribute extraction
- Training models to generate detailed furniture descriptions
- Developing automated furniture cataloging systems
- Benchmarking vision-language understanding capabilities

*Limitations*
- Training set consists of synthetic images which may not perfectly match real-world furniture
- Test set annotations are model-generated and may contain occasional inconsistencies
- Limited to four main furniture categories

### Pre-trained Models

**Qwen2-VL 2B LoRA Model**

A fine-tuned Qwen2-VL 2B model with LoRA adapters trained on the furniture synthetic dataset can be found at: [**furniture-caption-qwen-lora**](https://huggingface.co/filnow/furniture-caption-qwen-lora)

This model is specifically optimized for:
- Extracting structured furniture attributes from images
- Generating detailed furniture captions and descriptions
- Understanding furniture styles, materials, and design elements
- Providing consistent attribute classification across the four main furniture categories

The LoRA adapters provide efficient fine-tuning while maintaining the base model's general vision-language capabilities.

## Notes

- Experiments may require GPU access for optimal performance
- Some scripts may need additional model downloads or API keys
- Check individual files for specific requirements and usage instructions