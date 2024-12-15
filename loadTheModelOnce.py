from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

print('At the preprocessing step!! Wait!')

def load_everything():
	print("Importing the vision model")
	model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	print("vision model - DONE")

	print("Importing the Auto-Tokenizer")
	tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	print("AutoTokenizer - DONE")

	print("Importing the ViTImageProcessor")
	feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
	print('ViTImageProcessor - DONE')

	return model, tokenizer, feature_extractor