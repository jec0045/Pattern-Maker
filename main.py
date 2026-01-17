from PIL import Image

def crop_transparent_border(image_path, output_path):
    # Open the image
    image = Image.open(image_path)
    
    # Get the bounding box of non-transparent pixels
    bbox = image.getbbox()
    
    if bbox:
        # Crop the image to the bounding box
        cropped_image = image.crop(bbox)
        # Save the cropped image
        cropped_image.save(output_path)
        print(f"Cropped image saved to {output_path}")
    else:
        print("No non-transparent content found in the image.")

# Example usage
crop_transparent_border(r"Sample_Images\mudkip.png", "output.png")   