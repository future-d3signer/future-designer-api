import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat
import io
import base64
import requests

from app.models.model_provider import ModelProvider
from app.schemas.image_processing import ( # Import your Pydantic models
    DepthRequest, DepthResponse, StyleRequest, StyleResponse,
    CaptionRequest, CaptionResponse, ReplaceRequest,
    TransparencyRequest, TransparencyResponse, CompositeRequest
)
from app.utils.image_utils import ImageUtils # Assuming these are static methods
from app.utils.caption_utils import CaptionUtils
from app.utils.segmentation_utils import extract_furniture_segments_from_image  # Your existing function
from diffusers.image_processor import IPAdapterMaskProcessor
from vllm import SamplingParams


class ImageService:
    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
        # Pre-fetch models to avoid lazy loading during request, if preferred
        # self.model_provider.get_diffusion_pipelines() 
        # self.model_provider.get_depth_estimator()
        # ... etc. Or let them lazy load on first use.

    def generate_depth_map(self, source_image_b64: str) -> tuple[Image.Image, Image.Image, str]:
        image = ImageUtils.decode_image(source_image_b64).resize((1024, 1024))
        depth_estimator = self.model_provider.get_depth_estimator()
        depth_pil = depth_estimator(image)["depth"]
        depth_b64 = ImageUtils.encode_image(depth_pil)
        return image, depth_pil, depth_b64 # Return PIL images for potential chaining

    def generate_styled_image(self, style_key: str, depth_image_pil: Image.Image) -> str:
        prompts = self.model_provider.get_prompts()
        if style_key not in prompts:
            raise ValueError(f"Invalid style: {style_key}")

        pipeline_control, _ = self.model_provider.get_diffusion_pipelines()
        
        output = pipeline_control(
            prompt=prompts[style_key],
            negative_prompt=prompts["negative"],
            width=1024, height=1024,
            guidance_scale=1.5, num_inference_steps=7,
            control_image=[depth_image_pil], # This seems to be what you intended from original code
            controlnet_conditioning_scale=0.9, control_guidance_end=0.9,
            control_mode=[1], generator=torch.Generator(device="cuda"), eta=0.3,
            ip_adapter_image=self.model_provider.black_image,
        )
        generated_image_b64 = ImageUtils.encode_image(output.images[0])
        del output
        return generated_image_b64

    def generate_inpaint(self, base_prompt: str, depth_image_pil: Image.Image, 
                         orginal_image_pil:Image.Image, mask_image_pil: Image.Image) -> str:
        full_prompt = f"{base_prompt}, {self.model_provider.enhancement_prompt}"
        
        # Padded mask and blur logic from your original endpoint
        padded_mask = ImageUtils.add_mask_padding(mask_image_pil, padding=30)
        
        _, pipeline_inpaint = self.model_provider.get_diffusion_pipelines()
        # blured_image = pipeline_inpaint.mask_processor.blur(padded_mask, blur_factor=15)
        # The original code for blured_image seems to use a method not directly on pipeline_inpaint
        # Assuming mask_processor is accessible or you have a utility for it.
        # For now, let's assume padded_mask is used directly or you handle blur:
        # This line was: blured_image = model_manager.pipeline_inpaint.mask_processor.blur(padded_mask, blur_factor=15)
        # If pipeline_inpaint.mask_processor is not available, you'll need to adjust.
        # Let's assume IPAdapterMaskProcessor can be instantiated if needed, or that blur is done by ImageUtils
        processor = IPAdapterMaskProcessor() # Or get from pipeline if available
        blured_image = processor.blur(padded_mask, blur_factor=15)


        negative_prompt = "deformed, low quality, blurry, noise, grainy, duplicate, watermark, text, out of frame"
        seed = torch.randint(0, 100000, (1,)).item()

        output = pipeline_inpaint(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=orginal_image_pil,
            mask_image=blured_image, # This was `padded_mask` in original, ensure consistency
            num_inference_steps=5,
            control_image=[depth_image_pil],
            guidance_scale=2.5,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            controlnet_conditioning_scale=0.7,
            control_guidance_end=0.7,
            control_mode=[1],
            eta=0.3,
            strength=0.99,
            ip_adapter_image=self.model_provider.black_image
        )
        generated_image_b64 = ImageUtils.encode_image(output.images[0])
        del output
        return generated_image_b64

    def generate_delete(self, orginal_image_pil: Image.Image, box_image_pil: Image.Image) -> str:
        # ... (logic from your /generate_delete endpoint) ...
        # Remember to use self.model_provider to get models
        # e.g., _, pipeline_inpaint = self.model_provider.get_diffusion_pipelines()
        # The state of original_image_pil and depth_image_pil must be passed in.
        padded_mask = ImageUtils.add_mask_padding(box_image_pil, padding=64)

        blurred_mask = padded_mask.filter(ImageFilter.GaussianBlur(radius=15))  
        binary_mask_pil = blurred_mask.point(lambda x: 0 if x > 127 else 255) 

        original_array = np.array(orginal_image_pil)
        # Careful with binary_mask_pil for direct array comparison, ensure it's correctly formatted
        # Convert binary_mask_pil to a boolean mask array
        mask_array_for_zeroing = np.array(binary_mask_pil.convert("L")) == 0 # 0 is black (masked)
        
        original_array_copy = original_array.copy()
        original_array_copy[mask_array_for_zeroing] = [0, 0, 0] # Zero out masked area
        result_image_for_control = Image.fromarray(original_array_copy)

        mask_for_stats = padded_mask.convert("L")
        inverted_mask_for_stats = ImageOps.invert(mask_for_stats)
        stats = ImageStat.Stat(orginal_image_pil, mask=inverted_mask_for_stats)
        avg_color = tuple(int(c) for c in stats.mean[:3])
        neutral_fill = Image.new("RGB", orginal_image_pil.size, avg_color)
        neutral_image = Image.composite(orginal_image_pil, neutral_fill, binary_mask_pil) # binary_mask_pil should make unmasked areas transparent for composite

        negative_prompt = "furniture, objects, decorations, plants, clutter, people"
        generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 100000, (1,)).item())
        
        _, pipeline_inpaint = self.model_provider.get_diffusion_pipelines()

        output = pipeline_inpaint(
            prompt=self.model_provider.enhancement_prompt,
            negative_prompt=negative_prompt,
            image=neutral_image,  
            mask_image=blurred_mask, # The mask for inpainting
            control_image=[result_image_for_control],  
            control_mode=[7], # Check what mode 7 means
            num_inference_steps=8, guidance_scale=1.5, generator=generator,
            eta=0.3, strength=0.99, controlnet_conditioning_scale=1.0,  
            ip_adapter_image=self.model_provider.black_image
        )
        generated_image_b64 = ImageUtils.encode_image(output.images[0])
        del output
        return generated_image_b64


    def generate_replace(self, style_prompt: str, orginal_image_pil:Image.Image, 
                         mask_image_pil: Image.Image, adapter_image_name: str) -> str: # Pass original_image_pil and depth_image_pil
        # ... (logic from your /generate_replace endpoint) ...
        # Use self.model_provider, ImageUtils, etc.
        # State like original_image_pil and depth_image_pil must be passed in.
        response = requests.get(f"https://futuredesigner.blob.core.windows.net/futuredesigner1/{adapter_image_name}")
        response.raise_for_status()
        load_adapter_image = Image.open(io.BytesIO(response.content))
        
        full_prompt = f"{style_prompt}, {self.model_provider.enhancement_prompt}"

        padded_mask = ImageUtils.add_mask_padding(mask_image_pil, padding=30)

        processor = IPAdapterMaskProcessor()
        ip_masks = processor.preprocess(padded_mask, height=1024, width=1024)

        binary_mask_pil = padded_mask.point(lambda x: 0 if x > 127 else 255)
        
        original_array = np.array(orginal_image_pil)
        mask_array_for_zeroing = np.array(binary_mask_pil.convert("L")) == 0 # 0 is black (masked)
        
        original_array_copy = original_array.copy()
        original_array_copy[mask_array_for_zeroing] = [0, 0, 0]
        result_image_for_control = Image.fromarray(original_array_copy)

        negative_prompt = "deformed, low quality, blurry, noise, grainy, duplicate, watermark, text, out of frame"
        _, pipeline_inpaint = self.model_provider.get_diffusion_pipelines()

        output = pipeline_inpaint(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=orginal_image_pil,
            mask_image=padded_mask,
            num_inference_steps=5,
            guidance_scale=2.0,
            ip_adapter_image=load_adapter_image,
            generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 100000, (1,)).item()),
            strength=0.99,
            cross_attention_kwargs={"ip_adapter_masks": ip_masks},
            control_image=[result_image_for_control], # This uses the modified original image
            controlnet_conditioning_scale=1.0,
            eta=0.3,
            control_mode=[6] # Check what mode 6 means
        )
        generated_image_b64 = ImageUtils.encode_image(output.images[0])
        del output
        return generated_image_b64

    def generate_captions_for_image(self, source_image_b64: str) -> dict:
        image_pil = ImageUtils.decode_image(source_image_b64).resize((1024, 1024))
        
        sam_predictor = self.model_provider.get_sam_predictor() 
        dino_model = self.model_provider.get_dino_model()
        llm = self.model_provider.get_vlm_model()

        segmented_pil_list, raw_sam_masks, dino_boxes = extract_furniture_segments_from_image(
            image_pil, sam_predictor, dino_model
        )

        sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
        output_dict = {}

        for i, furniture_pil in enumerate(segmented_pil_list):
            furniture_b64 = ImageUtils.encode_image(furniture_pil)
            conversation = CaptionUtils.get_conversation_template(furniture_b64)
            
            x0, y0, x1, y1 = dino_boxes[i].int().cpu().numpy()
            box_image_np = np.zeros((1024, 1024), dtype=np.uint8)
            box_image_np[y0:y1, x0:x1] = 255
            
            mask_slice_np = raw_sam_masks[i, 0] # Assuming masks is a tensor
            mask_image_np = (mask_slice_np * 255).astype(np.uint8)
            
            mask_encoded = ImageUtils.encode_image(mask_image_np)
            box_encoded = ImageUtils.encode_image(box_image_np)
            
            # For VLLM, use generate, not chat, if it's a completion model
            # llm.chat might be specific to a VLLM setup with chat templates
            # For simplicity, assuming llm.chat is correct for your model
            llm_outputs = llm.chat(conversation, sampling_params=sampling_params)
            generated_text = llm_outputs[0].outputs[0].text
            del llm_outputs

            caption_data = CaptionUtils.parse_json_response(generated_text)
            
            # Assuming FurnitureItem schema is defined
            from app.schemas.image_processing import FurnitureItem # Place at top
            output_dict[f"furniture_{i}"] = FurnitureItem(
                caption=caption_data, # caption_data should match FurnitureDescription schema
                mask=mask_encoded,
                box=box_encoded,
                furniture_image=furniture_b64
            )
        return output_dict


    def make_furniture_transparent(self, furniture_image_url_suffix: str) -> str:
        # ... (logic from your /generate_transparency endpoint) ...
        # Use self.model_provider, ImageUtils
        response = requests.get(f"https://futuredesigner.blob.core.windows.net/futuredesigner1/{furniture_image_url_suffix}")
        response.raise_for_status()
        image_pil = Image.open(io.BytesIO(response.content))

        sam_predictor = self.model_provider.get_sam_predictor()
        dino_model = self.model_provider.get_dino_model()
        _, raw_sam_masks, _ = extract_furniture_segments_from_image(
            image_pil, sam_predictor, dino_model
        )
        
        mask_slice = raw_sam_masks[0, 0]
        rgba_image = image_pil.convert("RGBA")
        rgba_array = np.array(rgba_image)
        
        mask_array = mask_slice.astype(bool)
        rgba_array[:, :, 3] = np.where(mask_array, 255, 0)
        
        transparent_image_pil = Image.fromarray(rgba_array, mode="RGBA")
        transparent_image_b64 = ImageUtils.encode_image(transparent_image_pil, format="PNG")
        return transparent_image_b64

    def composite_and_blend_furniture(
        self, room_image_b64: str, furniture_image_b64: str, 
        position: dict, size: dict
    ) -> str:
        room_img = ImageUtils.decode_image(room_image_b64).convert("RGB")
        # furniture_image_b64 is already transparent PNG from client or previous step
        furniture_img = ImageUtils.decode_image(furniture_image_b64).convert("RGBA")
        
        furniture_resized = furniture_img.resize((size["width"], size["height"]), Image.Resampling.LANCZOS)
        
        initial_composite = room_img.copy()
        # Paste with transparency
        initial_composite.paste(furniture_resized, (position["x"], position["y"]), furniture_resized)
        
        # Create blend mask for diffusion
        blend_mask_pil = Image.new("L", room_img.size, 0)
        furniture_alpha_mask = furniture_resized.split()[3]
        blend_mask_pil.paste(furniture_alpha_mask, (position["x"], position["y"]))
        
        padded_blend_mask = ImageUtils.add_mask_padding(blend_mask_pil, padding=64)

        # Generate depth for the initial composite to guide blending
        depth_estimator = self.model_provider.get_depth_estimator()
        depth_pil = depth_estimator(initial_composite)["depth"]
        
        blend_prompt = "realistic lighting, seamless furniture integration, natural shadows, physically accurate placement"
        negative_prompt = "unrealistic lighting, floating furniture, harsh edges, artificial shadows"
        
        _, pipeline_inpaint = self.model_provider.get_diffusion_pipelines()
        output = pipeline_inpaint(
            prompt=blend_prompt, negative_prompt=negative_prompt,
            image=initial_composite, mask_image=padded_blend_mask,
            num_inference_steps=7, guidance_scale=1.5,
            control_image=[depth_pil], controlnet_conditioning_scale=0.7,
            control_guidance_end=0.7, control_mode=[1],
            generator=torch.Generator(device="cuda").manual_seed(torch.randint(0, 100000, (1,)).item()),
            strength=0.99, eta=0.3,
            ip_adapter_image=furniture_resized.convert("RGB"), 
        )
        result_img_pil = output.images[0]
        del output
        
        result_b64 = ImageUtils.encode_image(result_img_pil, format="PNG")
        # FastAPI will automatically make this JSON `{"composited_image": "data:image/png;base64,..."}`
        # if you return a Pydantic model. If returning raw dict, format it yourself.
        return f"data:image/png;base64,{result_b64}"