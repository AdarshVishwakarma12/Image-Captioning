from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

def process_uploaded_file(file_path, model, tokenizer, feature_extractor):
    # Simulate some processing

    path = file_path

    print('Sending Image to caption model!!!')

    # Set generation parameters
    gen_kwargs = {"max_length": 16, "num_beams": 4}

    def predict_step(image_paths):
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(model.device)

        output_ids = model.generate(pixel_values, **gen_kwargs)
        return [tokenizer.decode(output_id, skip_special_tokens=True).strip() for output_id in output_ids]

    # Run prediction on a sample image
    predictions = predict_step([path])

    print("the caption has been generated!")
    return predictions