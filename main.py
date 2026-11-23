
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes

import os
from utils import crop_transparent_border, resize_center_image, get_RGBA_list, \
    compare_colors, duplicate_checking, recolor_image, create_side_by_side, \
    add_grid, add_symbols, create_color_key, combine_image_and_key


# os.system('cls' if os.name == 'nt' else 'clear')

# Options
CROP = True
RESIZE = True

# ------------------------- Main Code -------------------------
base_path = r"C:\Users\jessc\OneDrive\Documents\Patterns"

# original_path = f"{base_path}\sunflower\sunflower_36_48.png"
# original_path = f"{base_path}\mushrooms\mushrooms_64_15.png"

original_path = r"C:\Users\jessc\OneDrive\Pictures\Cross Stitch\stardew_sprites\sprite_corn.png"

base_path = original_path.split(".")[0]

working_path = crop_transparent_border(original_path, base_path, CROP)
working_path = resize_center_image(working_path, base_path, RESIZE)

img_color_dict = get_RGBA_list(working_path)
# for color, count in img_color_dict.items():
#     print(count, "\t", color)

color_chart = compare_colors(img_color_dict)
print(color_chart)

working_path = recolor_image(working_path, base_path, color_chart)

# create_side_by_side(original_path, working_path, (0, 0, 0, 0))

grid_path = add_grid(working_path, base_path)

duplicate_checking(color_chart, original_path, grid_path)
color_key, symbol_path = add_symbols(working_path, grid_path, color_chart)

print(color_key)
key_path = create_color_key(color_key, grid_path)
combine_image_and_key(key_path, symbol_path)
