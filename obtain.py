from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import random

def load_everything():
    print("Importing the vision model")
    model = VisionEncoderDecoderModel.from_pretrained("./models")
    print("vision model - DONE")

    print("Importing the Auto-Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("./models")
    print("AutoTokenizer - DONE")

    print("Importing the ViTImageProcessor")
    feature_extractor = ViTImageProcessor.from_pretrained("./models")
    print('ViTImageProcessor - DONE')

    return model, tokenizer, feature_extractor

model, tokenizer, feature_extractor = load_everything()

# Set default generation parameters
default_gen_kwargs = {"max_length": 16, "num_beams": 4, "do_sample": True, "top_k": 50, "temperature": 0.7}

def predict_step(image_path, num_captions=5, gen_kwargs=None):
  
    if gen_kwargs is None:
        gen_kwargs = default_gen_kwargs

    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(model.device)

    captions = set()
    while len(captions) < num_captions:
        output_ids = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        captions.add(caption)
        print(captions)

    return captions

print(predict_step("./uploads/raceCar.jpeg"))