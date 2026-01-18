from PIL import Image
import numpy as np
import polars as pl
import pandas as pd
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from collections import Counter
import math
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes


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
    rgb_values = [img.getpixel((x, y)) for y in range(height) for x in range(width)]

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

    return  color_df


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


def closest_color(target, colors):
    tgt = rgb_to_lab([target[0], target[1], target[2]])

    best_color = None
    best_distance = float("inf")

    for color in colors:
        c = rgb_to_lab([color[0], color[1], color[2]])

        d = gpt_delta_e_ciede2000(tgt, c) # perceptual distance
        if d < best_distance:
            best_distance = d
            best_color = color

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
            h_bar_prime = (h1_prime + h2_prime + 360) / 2

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


# def compare_colors(img_colors, dmc_df):
#     dmc_colors = dmc_df["RGB"].to_list()  # Format as a list
#     # column_names = ["Original RGB", "Stitch Count", "Color Match RGB", "Floss Number", "Floss Name"]
#     # schema = {"Original RGBA": pl.String, "Stitch Count": pl.Int64, "Color Match RGB": pl.String,
#     #           "DMC Number": pl.Int64, "Floss Name": pl.String}

#     # Create an empty DataFrame from the schema
#     df = pl.DataFrame()

#     # Find color matches
#     for key, value in img_colors.items():
#         if key[3] > 0 and key[0] > 0:
#             rgb = [key[0], key[1], key[2]]
#             best_match = closest_color(rgb, dmc_colors)

#             # print(key, ".....", best_match)
#             # print(type(best_match))
#             best_match_R = best_match[0]
#             best_match_G = best_match[1]
#             best_match_B = best_match[2]
#             # print(best_match_R, best_match_G, best_match_B)

#             dmc_idx = dmc_df.index[(dmc_df["R"] == best_match_R) &
#                                    (dmc_df["G"] == best_match_G) &
#                                    (dmc_df["B"] == best_match_B)].astype(int)
#             dmc_num = dmc_df.loc[dmc_idx,'Floss']
#             dmc_name =  dmc_df.loc[dmc_idx,'DMC Name']

#             original_rgba = ' '.join(str(item) for item in key)
#             formatted_match = string_repr = np.array2string(best_match, precision=0, separator=', ')


#             # print(formatted_match)
#             # print(type(formatted_match))

#             new_row = pl.DataFrame({"Original RGBA": original_rgba, "Stitch Count": value,
#                                     "Color Match RGB": formatted_match,
#                                     "DMC Number": dmc_num, "Floss Name": dmc_name})
#             # df.extend(new_row)
#             df = pl.concat([df, new_row], ignore_index=True)
#     print(df)
#     return df

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
        if key[3] > 0 and key[0] > 0:
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



# ------------------------- Main Code -------------------------
# output_path = crop_transparent_border(r"Sample_Images\mudkip.png")
output_path = r"Sample_Images\mudkip_crop.png"
img_color_dict = get_RGBA_list(output_path)
dmc_colors_df = prep_dmc_color_lst()
# print(dmc_colors_df)

output = compare_colors(img_color_dict, dmc_colors_df)

print(output)
