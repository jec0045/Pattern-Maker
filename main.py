from PIL import Image
import numpy as np
import polars as pl
import pandas as pd
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from collections import Counter
import math
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes


from utils import crop_transparent_border, get_RGBA_list, prep_dmc_color_lst, \
    compare_colors


# ------------------------- Main Code -------------------------
# output_path = crop_transparent_border(r"Sample_Images\bulbasaur.png")
output_path = r"Sample_Images\mudkip_crop.png"
img_color_dict = get_RGBA_list(output_path)
dmc_colors_df = prep_dmc_color_lst()
output = compare_colors(img_color_dict, dmc_colors_df)

print(output)
