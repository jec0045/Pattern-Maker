
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes

from collections import defaultdict
from utils import files_and_folder_prep
from utils import crop_transparent_border, resize_center_image, \
    get_RGBA_list, compare_colors, duplicate_checking, recolor_image  #, \
#     create_side_by_side, add_grid, add_symbols, create_color_key, \
#     combine_image_and_key, colors_needed  # noqa: E501


# os.system('cls' if os.name == 'nt' else 'clear')

# Options
CROP = True
RESIZE = True
IMG_DEBUG = True

# ------------------------- Main Code -------------------------
BASE_PATH = r"C:\Users\jessc\OneDrive\Pictures\Cross Stitch\stardew_sprites"
DEBUG_PATH, files = files_and_folder_prep(BASE_PATH)

color_list = defaultdict(list)

for image in files:
    cropped_image = crop_transparent_border(image, CROP)
    resized_image = resize_center_image(cropped_image, RESIZE)

    img_color_dict = get_RGBA_list(resized_image)
    # for color, count in img_color_dict.items(): print(count, "\t", color)

    print(f"DEBUG: {len(img_color_dict):2d} unique RGB colors in {image}\n")

    continue

#     color_chart = compare_colors(img_color_dict)
#     # print(color_chart)

#     recolored_image = recolor_image(resized_image, color_chart)

#     gridded_img = add_grid(recolored_image)

#     if IMG_DEBUG:
#         create_side_by_side(
#             image, debug_path, resized_image, recolored_image, (0, 0, 0, 0))
#         duplicate_checking(color_chart, image, gridded_img, debug_path)

#     recolored_image.save(".test/recolored_image_output.png")
#     gridded_img.save(".test/gridded_img_output.png")

#     color_key, symboled_imgage = add_symbols(recolored_image, gridded_img, color_chart)  # noqa: E501
#     key_image = create_color_key(color_key)
#     combine_image_and_key(key_image, symboled_imgage, image)

#     # print("-"*100)
#     # print(image)
#     color_list = colors_needed(color_list, color_key, image)

# # # print(color_list)
# # for key, value in color_list.items():
# #     print(key,"\t", value)
