
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes

import os
import shutil
import tempfile
from PIL import Image
from pathlib import Path
from utils import crop_transparent_border, resize_center_image, get_RGBA_list, \
    compare_colors, duplicate_checking, recolor_image, create_side_by_side, \
    add_grid, add_symbols, create_color_key, combine_image_and_key, pprint


# os.system('cls' if os.name == 'nt' else 'clear')

# Options
CROP = True
RESIZE = True
IMG_DEBUG = True

# ------------------------- Main Code -------------------------
base_folder_path = r"C:\Users\jessc\OneDrive\Pictures\Cross Stitch\stardew_sprites"
files = ['C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_corn_test.png',]
#  'C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_corn.png',
#  'C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_hops.png',
#  'C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_melon.png',
#  'C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_starfruit.png',
#  'C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_summer_spangle.png',
#  'C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_tomato.png',
#  'C:\\Users\\jessc\\OneDrive\\Pictures\\Cross Stitch\\stardew_sprites\\sprite_wheat.png']

# Creates 'my_folder' and any missing parent folders safely
debug_path = f"{base_folder_path}\\debug"
Path(debug_path).mkdir(parents=True, exist_ok=True)

for image in files:

    cropped_image = crop_transparent_border(image, CROP)

    # pprint(f"DEBUG: {type(cropped_image)}")
    # pprint(f"DEBUG: {cropped_image}")

    resized_image = resize_center_image(cropped_image, RESIZE)

    # pprint(f"DEBUG: {type(resized_image)}")
    # pprint(f"DEBUG: {resized_image}")

    img_color_dict = get_RGBA_list(resized_image)
    # for color, count in img_color_dict.items(): print(count, "\t", color)

    color_chart = compare_colors(img_color_dict)
    print(color_chart)

    recolored_image = recolor_image(resized_image, color_chart)

    # pprint(f"DEBUG: {type(recolored_image)}")
    # pprint(f"DEBUG: {recolored_image}")

    gridded_img = add_grid(recolored_image)

    if IMG_DEBUG:
        create_side_by_side(
            image, debug_path, resized_image, recolored_image, (0, 0, 0, 0))
            # duplicate_checking(color_chart, original_path, grid_path)

    recolored_image.save(".test/recolored_image_output.png")
    gridded_img.save(".test/gridded_img_output.png")

    color_key, symboled_imgage = add_symbols(recolored_image, gridded_img, color_chart)
    key_image = create_color_key(color_key)
    combine_image_and_key(key_image, symboled_imgage, image)
