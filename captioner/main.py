# import os
# import sys
# from google import genai
# from google.genai import types

# # Set your Google API key
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# def generate_caption(filename):
#     """
#     Use Google's Generative AI to generate a caption based on image features.
#     """
#     file = client.files.upload(file=filename)
#     prompt = f"Generate a descriptive caption for the subject in following for training a FLux lora, avoid any other text other than caption"
#     response = client.models.generate_content(
#         model='gemini-2.0-flash-001', 
#         contents=[prompt, file],
#         config=types.GenerateContentConfig(
#             temperature=0,
#             max_output_tokens=100,
#     ),
#     )
#     return response.text.strip()

# def caption_images_in_directory(directory_path, output_file="captions.txt"):
#     """
#     Caption all images in a directory and save the captions to a file.
#     """
#     captions = []
#     for filename in os.listdir(directory_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             image_path = os.path.join(directory_path, filename)
#             print(f"Processing {filename}...")
            
#             # Generate a caption using Google's Generative AI
#             caption = generate_caption(image_path)
#             caption_file = image_path.split(".")[0] + ".txt"
#             with open(caption_file, "w") as f:
#                 f.write(caption)
    

# if __name__ == "__main__":
#     # Check if the directory path is provided as an argument
#     if len(sys.argv) < 2:
#         print("Usage: python script.py <directory_path>")
#         sys.exit(1)
    
#     # Get the directory path from the command-line arguments
#     image_directory = sys.argv[1]
    
#     # Check if the provided path is a valid directory
#     if not os.path.isdir(image_directory):
#         print(f"Error: {image_directory} is not a valid directory.")
#         sys.exit(1)
    
#     # Generate captions for images in the directory
#     caption_images_in_directory(image_directory)


import os
import sys
from google import genai
from google.genai import types

# Set your Google API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_captions_for_images(image_paths):
    """
    Use Google's Generative AI to generate captions for multiple images in one API call.
    """
    # Upload all image files
    # Create a single prompt with all uploaded files
    prompt = ["Generate a descriptive caption for each image below for training a Flux LoRA. Provide one caption per line, avoiding any other text"]
    for image_path in image_paths:
        prompt.append(client.files.upload(file=image_path))

    # Generate captions in one API call
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1000,  # Adjust token limit as needed
        ),
    )
    
    # Split the response text into individual captions
    captions = response.text.strip().split("\n")
    return captions


def caption_images_in_directory(directory_path):
    """
    Caption all images in a directory and save the captions to individual .txt files.
    """
    # Collect all image file paths
    image_paths = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    
    if not image_paths:
        print("No image files found in the directory.")
        return

    print(f"Processing {len(image_paths)} images...")

    # Generate captions for all images
    captions = generate_captions_for_images(image_paths)

    # Save each caption to a corresponding .txt file
    for image_path, caption in zip(image_paths, captions):
        caption_file = os.path.splitext(image_path)[0] + ".txt"
        with open(caption_file, "w") as f:
            f.write(caption)
        print(f"Caption saved to {caption_file}")


if __name__ == "__main__":
    # Check if the directory path is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    
    # Get the directory path from the command-line arguments
    image_directory = sys.argv[1]
    
    # Check if the provided path is a valid directory
    if not os.path.isdir(image_directory):
        print(f"Error: {image_directory} is not a valid directory.")
        sys.exit(1)
    
    # Generate captions for images in the directory
    caption_images_in_directory(image_directory)