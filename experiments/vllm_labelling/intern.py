import io
import json
import torch
import logging
import base64
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, Dataset, Features, Value, Image as ImageFeature
from huggingface_hub import login
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

@dataclass
class FurnitureMetadata:
    type: str
    style: str
    color: str
    material: str
    shape: str
    details: str
    room_type: str
    price_range: str

    @classmethod
    def get_empty(cls) -> 'FurnitureMetadata':
        return cls(**{field: "" for field in cls.__annotations__})

class DatasetProcessor:
    SYSTEM_PROMPT = """You are a furniture expert. Analyze images and provide descriptions in this exact JSON structure:
        {
            "type": "<must be one of: bed, chair, table, sofa>",
            "style": "<describe overall style>",
            "color": "<describe main color>",
            "material": "<describe primary material>",
            "shape": "<describe general shape>",
            "details": "<describe one decorative feature>",
            "room_type": "<specify room type>",
            "price_range": "<specify price range>"
        }
        Focus on maintaining this exact structure while providing relevant descriptions."""

    def __init__(self, model_name: str):
        self.logger = logging.getLogger(__name__)
        self.llm = self._initialize_llm(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
        self.sampling_params = SamplingParams(
            max_tokens=128,
            temperature=0.0,
            seed=42
        )

    @staticmethod
    def _initialize_llm(model_name: str) -> LLM:
        return LLM(
            model=model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            max_model_len=4096,
            max_num_seqs=5,
            seed=42,
        )

    @staticmethod
    def _parse_json_response(response: str) -> Optional[Dict]:
        try:
            if response.startswith("```json\n"):
                json_string = response.split("```json\n")[1].split("```")[0]
            else:
                json_string = response.replace('\n', '')
            return json.loads(json_string)
        except (json.JSONDecodeError, IndexError) as e:
            logging.error(f"JSON parsing error: {str(e)}")
            return None

    def _create_conversation(self, question: str) -> List[Dict]:
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        return prompt

    def process_image(self, image: Image.Image) -> FurnitureMetadata:
        try:
            conversation = self._create_conversation(self.SYSTEM_PROMPT)
            outputs = self.llm.generate({
                "prompt": conversation,
                "multi_modal_data": {"image": image}},
                sampling_params=self.sampling_params,
                use_tqdm=False)
            
            if json_data := self._parse_json_response(outputs[0].outputs[0].text):
                return FurnitureMetadata(**json_data)
            
            return FurnitureMetadata.get_empty()
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return FurnitureMetadata.get_empty()

    def process_dataset(self, dataset_name: str, split_name: str) -> Dataset:
        dataset = load_dataset(dataset_name)
        if split_name not in dataset:
            raise ValueError(f"Split {split_name} not found in dataset")

        data = dataset[split_name]
        processed_data = {field: [] for field in FurnitureMetadata.__annotations__}
        processed_data["image"] = []
        
        error_count = 0
        
        for item in tqdm(data):
            metadata = self.process_image(item['image'])
            if all(not value for value in vars(metadata).values()):
                error_count += 1
            
            processed_data["image"].append(item['image'])
            for field, value in vars(metadata).items():
                processed_data[field].append(value)

        self.logger.info(f"Processing completed with {error_count} errors")
        
        return Dataset.from_dict(
            processed_data,
            features=Features({
                "image": ImageFeature(),
                **{field: Value("string") for field in FurnitureMetadata.__annotations__}
            })
        )

def main(
    dataset_name: str,
    output_repo: str,
    split_name: str,
    hf_token: str,
    model_name: str = "OpenGVLab/InternVL2-8B"
) -> None:
    logging.basicConfig(level=logging.INFO)
    
    try:
        processor = DatasetProcessor(model_name)
        dataset = processor.process_dataset(dataset_name, split_name)
        
        login(token=hf_token)
        dataset.push_to_hub(output_repo)
        logging.info(f"Successfully pushed dataset to {output_repo}")
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main(
        dataset_name="filnow/furniture-synthetic-dataset",
        output_repo="filnow/futniture-intern-vl-8b",
        split_name="test"
    )

