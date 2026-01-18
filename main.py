
# pl.Config.set_tbl_rows(-1)  # Show all rows in any printed polars dataframes


from utils import crop_transparent_border, get_RGBA_list, prep_dmc_color_lst, \
    compare_colors, recolor_image, add_grid


# ------------------------- Main Code -------------------------
# crop_path = crop_transparent_border(r"Sample_Images\mudkip.png")
crop_path = crop_transparent_border(r"Sample_Images\TEST_MUDKIP_BLUE.png")
img_color_dict = get_RGBA_list(crop_path)
dmc_colors_df = prep_dmc_color_lst()
color_chart = compare_colors(img_color_dict, dmc_colors_df)
print(color_chart)
recolor_path = recolor_image(crop_path, color_chart)
# add_grid(recolor_path)