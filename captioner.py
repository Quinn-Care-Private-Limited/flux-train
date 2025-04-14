import os
from google import genai
from google.genai import types
from PIL import Image

# Set your Google API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def resize_image(image_path, target_size=512):
    """
    Resize the image so that the shortest side matches the target size while maintaining aspect ratio.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = target_size
            new_height = int((target_size / width) * height)
        else:
            new_height = target_size
            new_width = int((target_size / height) * width)
        
        resized_img = img.resize((new_width, new_height))
        resized_path = os.path.splitext(image_path)[0] + "_resized.jpg"
        resized_img.save(resized_path, "JPEG")
        return resized_path

def generate_caption(image_path):
    """
    Use Google's Generative AI to generate a caption based on image features.
    """
    resized_path = resize_image(image_path)
    file = client.files.upload(file=resized_path)
    if os.path.exists(resized_path):
        os.remove(resized_path)

    prompt = f"Generate a descriptive caption of subject for each image below for training a Flux LoRA. Provide detailed caption in one line, avoiding any other text"
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=[prompt, file],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=100,
    ),
    )
    return response.text.strip()

def caption_images_in_directory(dataset_dir):
    """
    Caption all images in a directory and save the captions to a file.
    """
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(dataset_dir, filename)
            print(f"Processing {filename}...")
            
            # Generate a caption using Google's Generative AI
            caption = generate_caption(image_path)
            caption_file = image_path.split(".")[0] + ".txt"
            with open(caption_file, "w") as f:
                f.write(caption)