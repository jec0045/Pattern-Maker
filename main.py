from PIL import Image
import numpy as np
import polars as pl
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from coloraide import Color
from collections import Counter
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes


def crop_transparent_border(image_path, output_path):
    image = Image.open(image_path)  # Open the image
    bbox = image.getbbox()  # Get the bounding box of non-transparent pixels

    if bbox:
        cropped_image = image.crop(bbox)  # Crop the image to the bounding box
        cropped_image.save(output_path)  # Save the cropped image
        print(f"Cropped image saved to {output_path}")
    else:
        print("No non-transparent content found in the image.")


def get_RGB_list(image_path):
    im = Image.open(r"Sample_Images\mudkip_crop.png").convert('RGB')  # Open and convert image to RGB
    na = np.array(im)  # Convert to NumPy array

    # Reshape array to list of RGB values and find unique colors
    unique_colors, counts = np.unique(na.reshape(-1, 3), axis=0, return_counts=True)

    unique_colors_list = unique_colors.tolist()
    count_list = counts.tolist()
    dataframe = pl.DataFrame({"Original Color RGB": unique_colors_list, "Occurrences": count_list})

    return dataframe


def get_RGBA_list(image_path):
    img = Image.open(image_path).convert('RGBA')

    # Extract RGB values for all pixels
    width, height = img.size
    rgb_values = [img.getpixel((x, y)) for y in range(height) for x in range(width)]

    rgb_counts = Counter(rgb_values)
    rgb_dict = dict(rgb_counts)
    
    return rgb_dict

    # # Filter out fully transparent pixels (alpha = 0)
    # non_transparent_pixels = img.get_flattened_data()  
    # colors = [color for color in non_transparent_pixels if color[3] > 0]
    # print(type(colors))
    # color_count = len(set(colors))


    # d = Counter(colors)
    # print("d", d)

    # print(dict(d))

    # print(f"Unique non-transparent colors: {color_count}")



def prep_dmc_color_lst():
    dmc_colors = pl.read_csv(  # Read in CSV & Format Columns
        "raw_threadcolors_data.csv", separator=",",ignore_errors=True,
        schema_overrides={"Floss": str, "R": int, "G": int, "B": int})
    dmc_colors = dmc_colors.with_columns(pl.concat_list("R", "G", "B").alias("RGB"))
    # Combine the "R", "G", and "B" columns into one column as a list

    # dmc_colors = dmc_colors.with_columns(
    #     pl.col("RGB")
    #     .map_elements(rgb_to_lab, return_dtype=pl.List(pl.Float64))
    #     .alias("Lab Color"))

    # dmc_colors_lst = dmc_colors["Lab Color"].to_list()
    # print(dmc_colors_lst)

    # dmc_colors_dict = {}
    # dmc_lab_colors_lst = []
    # for color in dmc_colors.iter_rows():
    #     lab_color = rgb_to_lab(color[6])  # Convert RGB to Lab Color
    #     dmc_colors_dict[color[0]] = lab_color  # Append dict with Floss and Lab Color
    #     dmc_lab_colors_lst.append(lab_color)
    #     # group = {color[0]: lab_color}  # Create a dict entry with the Floss Number and Lab Color

    # print(dmc_colors_dict)
    # print(dmc_lab_colors_lst)

    # return dmc_lab_colors_lst, dmc_colors_dict

    return dmc_colors["RGB"].to_list()


def rgb_to_lab(rgb):
    # lab = convert_color(sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255), LabColor)
    # return [lab.lab_l, lab.lab_a, lab.lab_b]
    # return convert_color(sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255), LabColor)

    # normalize to 0–1
    srgb = sRGBColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    lab = convert_color(srgb, LabColor)

    # ensure pure Python floats
    lab.lab_l = float(lab.lab_l)
    lab.lab_a = float(lab.lab_a)
    lab.lab_b = float(lab.lab_b)
    return lab


def closest_color_lab(target, colors):
    print(target)

    target_lab = rgb_to_lab(target)

    # print(target_lab)

    best_color = None
    best_distance = float("inf")

    for color in colors:
        d = delta_e_cie2000(target_lab, rgb_to_lab(color))
        if d < best_distance:
            best_distance = d
            best_color = color

    return best_color



def closest_color_coloraide(target, colors):
    tgt = Color(f"srgb({target[0]}, {target[1]}, {target[2]})").convert("lab")

    best_color = None
    best_distance = float("inf")

    # for color in colors:
    #     c = Color(f"srgb({color[0]}, {color[1]}, {color[2]})").convert("lab")
    #     d = tgt.delta_e(c)  # perceptual distance
    #     if d < best_distance:
    #         best_distance = d
    #         best_color = color

    return best_color



# ------------------------- Main Code -------------------------
# crop_transparent_border(r"Sample_Images\mudkip.png", r"Sample_Images\mudkip_crop.png")
# polars_dataframe = get_RGB_list(r"Sample_Images\mudkip_crop.png")
get_RGBA_list(r"Sample_Images\mudkip_crop.png")


# dmc_colors = prep_dmc_color_lst()
# for entry in polars_dataframe.iter_rows():
#     print(entry)
#     color = entry[0]
#     best_match = closest_color_coloraide(color, dmc_colors)
#     break
