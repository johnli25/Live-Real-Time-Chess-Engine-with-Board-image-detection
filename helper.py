import numpy as np
import cv2

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

def shi_tomasi_corners_to_chess_cells(corners):
    stack_corners = np.vstack(corners)

    # Sorting by x-coordinate
    sorted_corners = stack_corners[stack_corners[:, 0].argsort()]

    # Sorting every 9 coordinates by y-coordinate
    for i in range(0, len(sorted_corners), 9):
        sorted_corners[i:i+9] = sorted_corners[i:i+9][sorted_corners[i:i+9][:, 1].argsort()]
    return sorted_corners

def chessboard_corners_to_chess_cells(corners):
    sorted_corners = sorted(corners, key=lambda x: x[0][0])
    sorted_corners = sorted(sorted_corners, key=lambda x: x[0][1])
    return sorted_corners

def transform(img, corners):

    # reformat corners by vertical stacking 
    all_corners = np.vstack(corners)
    print(all_corners.shape)

    topleft_idx = np.argmin(all_corners[:9,1])
    # topleft = all_corners[topleft_idx]
    topleft = all_corners[0]

    topright_idx = np.argmax(all_corners[9:17,1])
    # topright = all_corners[topright_idx]
    topright = all_corners[8]

    bottomleft_idx = np.argmin(all_corners[72:81,1])
    # bottomleft = all_corners[bottomleft_idx + 72]
    bottomleft = all_corners[72]

    bottomright_idx = np.argmax(all_corners[72:81,1])
    # bottomright = all_corners[bottomright_idx + 72]
    bottomright = all_corners[80]

    print("topleft, topright, bottomleft, bottomright", topleft, topright, bottomleft, bottomright)
    src = np.float32([topleft, topright, bottomleft, bottomright])
    dst = np.float32([[0,0], [800,0], [0,800], [800,800]])

    M = cv2.getPerspectiveTransform(src, dst)
    print("M", M)
    warped_img = cv2.warpPerspective(img, M, (800, 800))

    cell_size = warped_img.shape[0] // 8

    for corner in corners:
        x, y = corner[0]
        transformed_corner = cv2.perspectiveTransform(np.array([[x, y]], dtype=np.float32), M)

        col_idx = int(transformed_corner[0][0] // cell_size)
        row_idx = int(transformed_corner[0][1] // cell_size)

        col_label = chr(ord('a') + col_idx)
        row_label = 8 - row_idx

        chess_coord = col_label + str(row_label)
        print("chess_coord", chess_coord)


