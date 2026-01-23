
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes

import os
from utils import crop_transparent_border, get_RGBA_list, prep_dmc_color_lst, \
    compare_colors, duplicate_checking, recolor_image, create_side_by_side, \
    add_grid, add_symbols, create_color_key, combine_image_and_key


os.system('cls' if os.name == 'nt' else 'clear')

# ------------------------- Main Code -------------------------
crop_path = crop_transparent_border(r"Sample_Images\mudkip\mudkip.png")
img_color_dict = get_RGBA_list(crop_path)
dmc_colors_df = prep_dmc_color_lst()
color_chart = compare_colors(img_color_dict, dmc_colors_df)
recolor_path = recolor_image(crop_path, color_chart)
# create_side_by_side(crop_path, recolor_path)
grid_path = add_grid(recolor_path)
# duplicate_checking(color_chart, crop_path, grid_path)
color_key, symbol_path = add_symbols(recolor_path, grid_path, color_chart)
key_path = create_color_key(color_key, grid_path)
combine_image_and_key(key_path, symbol_path)
