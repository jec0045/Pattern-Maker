from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
import polars as pl
import pandas as pd
import json
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colorspacious import cspace_convert
from collections import Counter
import math
import random

# Constants
SCALE = 64


# -------------------------------------------------------------- Called By Main
def crop_transparent_border(image_path):
    image = Image.open(image_path)  # Open the image
    bbox = image.getbbox()  # Get the bounding box of non-transparent pixels
    output_path = image_path  # Default the output to the original image

    if bbox:
        output_path = output_path.replace(".png", "_crop.png")
        cropped_image = image.crop(bbox)  # Crop the image to the bounding box
        cropped_image.save(output_path)  # Save the cropped image
        print(f"Cropped image saved to {output_path}")
    else:
        print("No non-transparent content found in the image.")

    return output_path


def get_RGBA_list(image_path):
    img = Image.open(image_path).convert('RGBA')

    width, height = img.size  # Extract RGB values for all pixels
    rgb_values =[
        img.getpixel((x, y)) for y in range(height) for x in range(width)]

    rgb_counts = Counter(rgb_values)
    rgb_dict = dict(rgb_counts)

    sorted_dict = dict(sorted(rgb_dict.items()))

    return sorted_dict


def prep_dmc_color_lst():
    dmc_colors = pl.read_csv(  # Read in CSV & Format Columns
        "raw_threadcolors_data.csv", separator=",",ignore_errors=True,
        schema_overrides={"Floss": str, "R": int, "G": int, "B": int})
    dmc_colors = dmc_colors.with_columns(pl.concat_list("R", "G", "B").alias("RGB"))
    # Combine the "R", "G", and "B" columns into one column as a list

    color_df = dmc_colors.to_pandas()  # Format as a pandas dataframe
    color_df['Floss'] = color_df['Floss'].astype(str).str.zfill(4)

    return color_df


def compare_colors(img_colors, dmc_df):
    dmc_colors = dmc_df["RGB"].to_list()
    
    df = pd.DataFrame(columns=[
        "Original RGBA",
        "Stitch Count",
        "Color Match RGB",
        "DMC Number",
        "Floss Name"
    ])

    for key, value in img_colors.items():
        if key[3] > 0:  # and key[0] > 0:
            rgb = [key[0], key[1], key[2]]
            best_match = closest_color(rgb, dmc_colors)

            best_match_R, best_match_G, best_match_B = best_match

            match_rows = dmc_df[
                (dmc_df["R"] == best_match_R) &
                (dmc_df["G"] == best_match_G) &
                (dmc_df["B"] == best_match_B)
            ]

            if match_rows.empty:
                continue

            dmc_num = match_rows.iloc[0]["Floss"]
            dmc_name = match_rows.iloc[0]["DMC Name"]

            original_rgba = ' '.join(map(str, key))
            formatted_match = np.array2string(
                np.array(best_match),
                precision=0,
                separator=', '
            )

            df.loc[len(df)] = [
                original_rgba,
                value,
                formatted_match,
                dmc_num,
                dmc_name
            ]

    return df


def recolor_image(image_path, pandas_colors):
    # Load and convert image to RGBA
    img = Image.open(image_path).convert('RGBA')
    data = np.array(img)
    color_map = {}

    for row in pandas_colors.iterrows():
        # Get Original and New Color
        original = row[1]["Original RGBA"].split()
        # print("OG :", type(original), "...", original)
        new = (row[1]["Color Match RGB"].replace(",", "").replace("[", "")
               .replace("]", "").split())
        new.extend(['255'])
        # print("NEW:", type(new), "...", new)

        # Convert Colors from Str to Int
        original_int = tuple(list(map(int, original)))
        new_int = tuple(list(map(int, new)))

        # Define color mappings: old_color -> new_color
        color_map[original_int] = new_int


    # Replace each color in the map
    for old_color, new_color in color_map.items():
        # Create mask for pixels matching the old color
        mask = np.all(data == old_color, axis=-1)
        data[mask] = new_color

    # Convert back to PIL Image and save
    output_path = image_path.replace("crop.png", "recolor.png")
    result_img = Image.fromarray(data)
    result_img.save(output_path)
    print(f"Recolored image saved to {output_path}")

    return output_path


def duplicate_checking(color_chart, original_path, scaled_path):
    # Find Color Duplicates
    color_list = color_chart['DMC Number'].tolist()
    counts = Counter(color_list)
    duplicates = [item for item, count in counts.items() if count > 1]

    # Get the Base Image loaded as a refrence
    base = Image.open(original_path).convert("RGBA")
    w, h = base.size

    # Open the scalled/recolored/gridded Image
    scaled = Image.open(scaled_path).convert("RGBA")

    symbol_list = get_symbol_list()

    # Testing
    for color in duplicates:
        selected_rows = color_chart[color_chart['DMC Number'] == color]
        for idx, row in selected_rows.iterrows():
            target_str = row["Original RGBA"]
            target_color = tuple(map(int, tuple(target_str.split())))
            print(target_color)

            symbol = Image.open(symbol_list[idx]).convert("RGBA")
            symbol = symbol.resize((SCALE, SCALE))

            base_pixels = base.load()

            for y in range(h):
                for x in range(w):
                    if base_pixels[x, y] == target_color:
                        px = x * SCALE
                        py = y * SCALE

                        scaled.paste(symbol, (px, py), symbol)

    scaled.show()


def create_side_by_side(img1_path, img2_path):
    """
    Combines two images side-by-side using the Pillow library.
    """
    # Open the images
    img1 = Image.open(img1_path).convert("RGBA")
    img2 = Image.open(img2_path).convert("RGBA")

    # Calculate the total width of the new image
    total_width = img1.width + img2.width
    total_height = img1.height # Both heights are now the same

    # Create a new transparent image with the combined width and consistent height
    # Mode "RGBA" is crucial for transparency, and color (0,0,0,0) makes it fully transparent
    combined_img = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))

    # Paste the first image onto the new image at coordinates (0, 0)
    combined_img.paste(img1, (0, 0), mask=img1) # Use mask to maintain transparency

    # Paste the second image next to the first one, starting at the first image's width
    combined_img.paste(img2, (img1.width, 0), mask=img2) # Use mask to maintain transparency

    # Save the result
    output_path = img1_path.replace("_crop.png", "_side_by_side.png")
    combined_img.save(output_path)
    print(f"Images combined and saved to {output_path}")


def add_grid(image_path):
    # Refrence: https://randomgeekery.org/post/2017/11/drawing-grids-with-python-and-pillow/

    # Open Image
    img = Image.open(image_path).convert("RGBA")

    # Create a temp background for testing
    temp_bckgnd_color = (245, 242, 208)
    background = Image.new("RGBA", img.size, temp_bckgnd_color)
    background.paste(img, (0, 0), img)
    img = background

    # Scale up using nearest-neighbor (e.g., 2x)
    scaled_img = img.resize((img.width * SCALE, img.height * SCALE), Image.Resampling.NEAREST)
    image = scaled_img

    # Image Sizing & Step Size
    x_start = 0
    x_end = image.width
    y_start = 0
    y_end = image.height
    step_size = SCALE
    box_size = 10

    # Set Colors
    mionr_line = "grey"
    major_line = "black"

    # Draw pixel lines (vertical)
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=mionr_line, width=1)

    # Draw pixel lines (horizontal)
    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=mionr_line, width=1)

    # Draw Box around entrie image
    bbox_coordinates = (0, 0, image.width - 1, image.height - 1)
    draw.rectangle(bbox_coordinates, outline=mionr_line)

    # Draw 10 x 10 Boxes
    for x in range(0, image.width, step_size*box_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=major_line, width=6)
    for y in range(0, image.height, step_size*box_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=major_line, width=6)

    # Draw a vertical line in the middle of the image
    x = image.width / 2
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill="purple", width=3)

    # Draw a horizontal line in the middle of the image
    y = image.height / 2
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill="purple", width=3)

    # Finish Drawing
    del draw

    output_path = image_path.replace("_recolor.png", "_grid.png")
    image.save(output_path)
    print(f"Gridded image saved to {output_path}")

    return output_path


def add_symbols(base_image_path, grid_image_path, color_chart):

    # Get the Base Image loaded as a refrence
    base = Image.open(base_image_path).convert("RGBA")
    w, h = base.size
    base_pixels = base.load()

    # Open the scalled/recolored/gridded Image
    scaled = Image.open(grid_image_path).convert("RGBA")

    # Symbol List
    symbol_list = get_symbol_list()

    # Handle Pandas Dataframe & Remove Duplicates
    color_key = color_chart[['DMC Number', 'Floss Name',
                             'Stitch Count', 'Color Match RGB']]
    color_key = color_key.groupby("Floss Name", as_index=False).agg({
        'DMC Number': 'first',
        'Stitch Count': 'sum',
        'Color Match RGB': 'first'}).sort_values(by='DMC Number')
    color_key = color_key.reset_index(drop=True)
    color_key['Symbol'] = ""

    # Apply Symbols
    for idx, (_, row) in enumerate(color_key.iterrows()):
        target_color = rgb_to_rgba(row["Color Match RGB"])
        color_correct = color_correction(target_color)
        # print(idx, target_color, "Color Correction: ", color_correct)

        if idx < len(symbol_list):
            # print(idx, len(symbol_list))
            symbol = Image.open(symbol_list[idx]).convert("RGBA")
            symbol = symbol.resize((SCALE, SCALE))
            color_key.loc[idx, 'Symbol'] = symbol_list[idx]

            if color_correct:
                arr = np.array(symbol)
                arr[..., :3] = 255 - arr[..., :3]
                symbol = Image.fromarray(arr, "RGBA")

            for y in range(h):
                for x in range(w):
                    # if base_pixels[x, y] == target_color:
                    if base_pixels[x, y][:4] == target_color[:4]:
                        px = x * SCALE
                        py = y * SCALE

                        scaled.paste(symbol, (px, py), symbol)

    output_path = grid_image_path.replace("_grid.png", "_icons.png")
    scaled.save(output_path)
    print(f"Image w/ symbols/icons saved to {output_path}")

    return color_key, output_path


def create_color_key(key, image_path):
    key = key.sort_values(by='DMC Number').reset_index(drop=True)

    # Create an Image
    height = len(key) * SCALE
    width = 900
    image = Image.new("RGB", size=(width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Image Sizing & Step Size
    x_start = 0
    x_end = image.width
    step_size = SCALE
    margin = 5
    font_size = SCALE - (4*margin)

    # Cycle Through and Draw Key
    for y in range(0, image.height, step_size):
        index = int(y/SCALE)

        # Draw Horizontal Lines
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill="black", width=1)

        # Draw a Colored Square
        color_str = (key["Color Match RGB"].iloc[index].replace('[', '')
                     .replace(']', '').replace(',', ''))
        color = tuple(map(int, tuple(color_str.split())))

        square_coords = [(x_start+margin, y+margin),
                         (x_start+SCALE-margin, y+SCALE-margin)]
        draw.rectangle(square_coords, fill=color)

        # Symbols
        symbol_path = key["Symbol"].iloc[index]
        # print(color, symbol_path)
        if symbol_path:
            symbol_scale = SCALE - (margin * 2)
            symbol = Image.open(symbol_path).convert("RGBA")
            symbol = symbol.resize((symbol_scale, symbol_scale))

            if color_correction(color):
                arr = np.array(symbol)
                arr[..., :3] = 255 - arr[..., :3]
                symbol = Image.fromarray(arr, "RGBA")

            image.paste(symbol, (x_start+margin, y+margin), symbol)

        # Labels
        dmc_num = key["DMC Number"].iloc[index]
        dmc_name = key["Floss Name"].iloc[index]

        try:
            # Load a TrueType font file and set the size
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            print("Font file not found, using default font which is not scalable.")
            font = ImageFont.load_default() 

        text_start = x_start+SCALE+(2*margin)
        draw.text((text_start, y+margin), f"DMC {dmc_num} - {dmc_name}",
                  fill="black", font=font)

    output_path = image_path.replace("_grid.png", "_key.png")
    image.save(output_path)
    print(f"Cross stitch key saved to {output_path}")

    return output_path


def combine_image_and_key(key_path, image_path):

    # Get Image Dimensions
    img = Image.open(image_path).convert("RGBA")
    img_w, img_h = img.size

    # Get Key Dimensions
    key = Image.open(key_path).convert("RGBA")
    key_w, key_h = key.size

    max_h = max(img_h, key_h)
    sum_w = img_w + key_w

    canvas = Image.new("RGB", size=(sum_w, max_h), color=(255, 255, 255))
    # canvas = ImageDraw.Draw(image)
    canvas.paste(img, (0, 0), mask=img) # Use mask to maintain transparency
    canvas.paste(key, (img_w, 0), mask=key) # Use mask to maintain transparency

    output_path = image_path.replace("_icons.png", "_FINAL.png")
    canvas.save(output_path)
    print(f"Final image saved to {output_path}")
    canvas.save(output_path)


# -------------------------------------------------------- Supporting Functions
def rgb_to_lab(rgb):
    # lab = convert_color(sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255), LabColor)
    # return [lab.lab_l, lab.lab_a, lab.lab_b]
    # return convert_color(sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255), LabColor)

    # # normalize to 0–1
    # srgb = sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    # lab = convert_color(srgb, LabColor)

    srgb = sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, is_upscaled=False)
    lab = convert_color(
        srgb,
        LabColor,
        target_illuminant="d65"
    )

    # ensure pure Python floats
    lab.lab_l = float(lab.lab_l)
    lab.lab_a = float(lab.lab_a)
    lab.lab_b = float(lab.lab_b)
    return lab


def closest_color(target, colors):
    # tgt = rgb_to_lab([target[0], target[1], target[2]])

    best_color = None
    best_distance = float("inf")

    for color in colors:
        # c = rgb_to_lab([color[0], color[1], color[2]])
        d = cam02_ucs_distance(target, color)  # perceptual distance

        if d < best_distance:
            best_distance = d
            best_color = color
            # print(f"NEW BEST: {color} \t {d}")
        # else:
        #     print("         ", color, "   \t", d)

    return best_color


def gpt_delta_e_ciede2000(color1, color2):
    """
    Calculate CIEDE2000 color difference between two Lab colors.
    lab1, lab2: tuples (L, a, b)
    Returns: ΔE00
    """

    L1 = color1.lab_l
    a1 = color1.lab_a
    b1 = color1.lab_b
    L2 = color2.lab_l
    a2 = color2.lab_a
    b2 = color2.lab_b

    # Step 1: Chroma
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2

    # Step 2: G factor
    C_bar7 = C_bar**7
    G = 0.5 * (1 - math.sqrt(C_bar7 / (C_bar7 + 25**7)))

    # Step 3: Adjusted a values
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    # Step 4: Adjusted chroma
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    C_bar_prime = (C1_prime + C2_prime) / 2

    # Step 5: Hue angles (degrees)
    def hue_angle(a, b):
        angle = math.degrees(math.atan2(b, a))
        return angle if angle >= 0 else angle + 360

    h1_prime = hue_angle(a1_prime, b1)
    h2_prime = hue_angle(a2_prime, b2)

    # Step 6: Differences
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        dh = h2_prime - h1_prime
        if abs(dh) <= 180:
            delta_h_prime = dh
        elif dh > 180:
            delta_h_prime = dh - 360
        else:
            delta_h_prime = dh + 360

    delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(
        math.radians(delta_h_prime / 2)
    )

    # Step 7: Averages
    L_bar_prime = (L1 + L2) / 2

    if C1_prime * C2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            h_bar_prime = (h1_prime + h2_prime) / 2
        else:
            if (h1_prime + h2_prime) < 360:
                h_bar_prime = (h1_prime + h2_prime + 360) / 2
            else:
                h_bar_prime = (h1_prime + h2_prime - 360) / 2


    # Step 8: Weighting functions
    S_L = 1 + (0.015 * (L_bar_prime - 50)**2) / math.sqrt(
        20 + (L_bar_prime - 50)**2
    )

    S_C = 1 + 0.045 * C_bar_prime

    T = (
        1
        - 0.17 * math.cos(math.radians(h_bar_prime - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar_prime))
        + 0.32 * math.cos(math.radians(3 * h_bar_prime + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar_prime - 63))
    )

    S_H = 1 + 0.015 * C_bar_prime * T

    # Step 9: Rotation term
    delta_theta = 30 * math.exp(-((h_bar_prime - 275) / 25)**2)
    R_C = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta))

    # Step 10: Final ΔE00
    delta_E = math.sqrt(
        (delta_L_prime / S_L)**2
        + (delta_C_prime / S_C)**2
        + (delta_H_prime / S_H)**2
        + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )

    return delta_E


def cam02_ucs_distance(rgb1, rgb2):
    """
    Perceptual distance using CAM02-UCS (J′a′b′).
    rgb1, rgb2: [R, G, B] in 0–255
    Returns: Euclidean distance in CAM02-UCS
    """

    # normalize to 0–1
    rgb1 = np.array(rgb1) / 255.0
    rgb2 = np.array(rgb2) / 255.0

    cam1 = cspace_convert(rgb1, "sRGB1", "CAM02-UCS")
    cam2 = cspace_convert(rgb2, "sRGB1", "CAM02-UCS")

    return np.linalg.norm(cam1 - cam2)


def get_symbol_list():
    # Get JSON of avaliable symbols
    with open(r'Symbols\symbols.json', 'r') as file:
    # Use json.load() to deserialize the file content into a Python list
        my_list_str = json.load(file)

    my_list_str = my_list_str.replace('[', '').replace(']', '').replace('"', '')
    # print(my_list_str)
    my_list = my_list_str.split(', ')

    random.shuffle(my_list)

    return my_list


def rgb_to_rgba(color):
    target_color = (color.replace(",", "").replace("[", "")
                    .replace("]", "").split())
    target_color.extend(['255'])
    int_list = [int(x) for x in target_color]
    target_tuple = tuple(int_list)
    return target_tuple


def color_correction(color):
    """
    Start with all symbols in black, check if the symbol should be changed to white.
    Return True is the symbol needs to be changed to white, return False if it is good as is.
    """

    r, g, b = color[:3]

    # Relative luminance (0 = black, 255 = white)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return False if luminance > 127.5 else True

