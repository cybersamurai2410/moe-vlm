import base64
import requests
from PIL import Image
from io import BytesIO
from sagemaker.huggingface.model import HuggingFacePredictor

# Function to encode image from URL to base64
def encode_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Initialize SageMaker predictor (replace with your real endpoint name)
predictor = HuggingFacePredictor(endpoint_name="ENDPOINT")

# Define your prompt
question = "What is shown in this image, and what is the relevance for materials design? Include a discussion of multi-agent AI."
prompt = f"<|image_1|>\n{question}"

# Image URL to encode
image_url = "IMAGE-URL"
image_base64 = encode_image_from_url(image_url)

# Payload sent to SageMaker endpoint
payload = {
    "inputs": prompt,
    "image": image_base64,
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "do_sample": True,
        "stop_strings": ['<|end|>', '<|endoftext|>']
    }
}

# Send request to SageMaker model endpoint
response = predictor.predict(payload)

# Print the model response
print("MoE Model Response:", response)
