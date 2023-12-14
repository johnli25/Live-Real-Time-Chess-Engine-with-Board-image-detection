# heic_files = os.listdir(heic_directory)
# def heic_to_jpg(heic_directory, output_directory):
#     for filename in heic_files:
#         print("directory + filename:", os.path.join(heic_directory, filename))
#         heif_file = pillow_heif.read_heif(os.path.join(heic_directory, filename))
#         image = Image.frombytes(
#             heif_file.mode,
#             heif_file.size,
#             heif_file.data,
#             "raw",
#         )
#         image.save(os.path.join(output_directory, filename.replace(".HEIC", ".jpg")), format="jpeg")

def crop_image(img, corners):
    min_x = min(corners[:,0,0])
    max_x = max(corners[:,0,0])
    min_y = min(corners[:,0,1])
    max_y = max(corners[:,0,1])
    cell_length, cell_width = (max_x - min_x) / 6, (max_y - min_y) / 6
    cell_length_width = min(cell_length, cell_width)

    print("original min_x, max_x, min_y, max_y", min_x, max_x, min_y, max_y)
    # calculate new min and max x and y 
    min_x, max_x = min_x - cell_length_width, max_x + cell_length_width
    min_y, max_y = min_y - cell_length_width, max_y + cell_length_width
    print("new min_x, max_x, min_y, max_y", min_x, max_x, min_y, max_y)

    # crop image
    cropped_img = img[int(min_y):int(max_y), int(min_x):int(max_x)]
    print("cropped_img.shape", cropped_img.shape)

    # adjust corners
    corners[:,0,0] = corners[:,0,0] - min_x
    corners[:,0,1] = corners[:,0,1] - min_y
    return cropped_img, corners

