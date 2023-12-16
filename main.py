import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import stockfish
import chess
import chess.svg
import chess.engine
import os 
from PIL import Image
import pillow_heif
import helper
from roboflow import Roboflow
rf = Roboflow(api_key="nNDxYmDvU1b7mbzgcwHV")
project = rf.workspace().project("chess-13fgy")
model = project.version(1).model

engine_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"
np.set_printoptions(threshold=np.inf)

def returnBoardConfigurationFromImage(img_path):
    board_8x8 = np.array([['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
                              ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0],
                              ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                              ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']]) # temporary test board
    return board_8x8

def convert_rowcol_to_algebraic_notation(row, col):
    file = chr(ord('a') + col)
    rank = str(8 - row)
    # print(file + rank)
    return file + rank

def set_piece_on_board(board, piece, position):
    piece = chess.Piece.from_symbol(piece)
    board.set_piece_at(chess.parse_square(position), piece)

# inputs: board_img_nparray: 8x8 numpy array of the board
def returnNextBestMove(board_img_nparray): 
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        # board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        board = chess.Board()
        board.clear()
        for i in range(len(board_img_nparray)):
            for j in range(len(board_img_nparray[0])):
                piece = board_img_nparray[i, j]
                if piece != '0':
                    chess_pos_algebraic_notation = convert_rowcol_to_algebraic_notation(i, j)
                    set_piece_on_board(board, piece, chess_pos_algebraic_notation)
        print(board)

        info = engine.analyse(board, chess.engine.Limit(time=1.0))

        recommened_move = info["pv"][0]
        return recommened_move


# main driver code:
# current_8x8_board = returnBoardConfigurationFromImage("test.png")
# print(returnNextBestMove(current_8x8_board))

heic_directory = "./chess-images-training-dataset-heic/"
jpg_directory = './chess-images-training-dataset-jpg/'

def find_chessboard_corners(img):
    # convert img to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(img_gray, (7,7), None)
    return corners

test_corners = find_chessboard_corners(cv2.imread("./online_walnut_chess.jpg"))
print("online walnut chess test_corners", test_corners)
cv2.drawChessboardCorners(cv2.imread("./online_walnut_chess.jpg"), (7,7), test_corners, True)

def random_color():
    print("random color", np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

def canny_edge_detection(img):
    # convert img to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 200, 400, apertureSize=5) # apertureSize is the size of Sobel kernel (3,5,7) used for find image gradients, 100 = min threshold for hysterisis, 200 = max threshold for hysterisis procedure

    # find contours: contours returns a list of contours, hierarchy describes the child-parent relationships between contours (e.g. if one contour is inside another contour)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    # sort countours by descending area: internally, it temporarily convert array of points to enclosed area and determine closed area using cv2.contourArea
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # for i, contour in enumerate(contours):
    #     cv2.drawContours(img, contour, -1, random_color(), 2) # arguments are: image, contours, contourIdx (-1 means draw all contours), color, thickness
    # plt.imshow(img)
    # plt.suptitle("canny edge detection + draw all contours")
    # plt.show()

    chessboard_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # extract perimeter of contour

        # approximate the contour to a polygon by simplifying contour shape to another shape with fewer vertices
        approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True) # approx is a list of points representing the polygon
        print("len approx", len(approx))
        if len(approx) == 4:
            chessboard_contour = approx
            break

    if chessboard_contour is not None:
        print("chessboard contour len", len(chessboard_contour))
        # draw contours on image
        for contour in chessboard_contour:
            cv2.circle(img, tuple(contour[0]), 10, (0, 0, 255), -1) # pin point the corners of the chessboard in red
        cv2.drawContours(img, [chessboard_contour], -1, (0, 255, 0), 2) # arguments are: image, contours, contourIdx (-1 means draw all contours), color, thickness
        # cv2.imshow("chessboard contours", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.imshow(img)
        plt.suptitle("chessboard contours")
        plt.show()

    else:
        print("chessboard contours not found")

def to_txt_file(name, array):
    with open(name + ".txt", "w") as f:
        f.write(name + "\n")
        f.write(str(array))

output_directory = [jpg_directory + "IMG_5090.jpg"] # , jpg_directory + "IMG_5091.jpg", jpg_directory + "IMG_5092.jpg"]
# output_directory = os.listdir(jpg_directory) 
# output_directory = ['./manually_cropped_chessboard.png']
output_directory = ['./IMG_5096.jpg']
for img_path in output_directory:
    # img_path = jpg_directory + img_path
    img = cv2.imread(img_path)

    # apply Gaussian blur to reduce noise
    # img = cv2.GaussianBlur(img, (5,5), 0)

    # downscale image:
    img = cv2.resize(img, (0,0), fx=0.5 ** 3, fy=0.5 ** 3)

    # Shi Tomasi Corner Detection (pretty good)
    corners2 = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 81, 0.1, 10) # args are: image, maxCorners, qualityLevel, minDistance
    corners2_array = np.array(corners2).astype(np.int32)
    for corner in corners2_array:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 1, [255, 255, 0], -1)
    print("length of corners2_array (Shi Tomasi)=", len(corners2_array))
    sorted_shi_tomasi_corners = helper.shi_tomasi_corners_to_chess_cells(corners2_array)
    to_txt_file("Shi_Tomasi_corners2", sorted_shi_tomasi_corners)
    # helper.transform(img, sorted_shi_tomasi_corners)

    # Harris Corner Detection (pretty bad)
    print("img shape", img.shape)
    corners3_vals = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04) # args are: image, blockSize, ksize, k
    corners3_vals = cv2.dilate(corners3_vals, None)
    corners3_coords = []

    for i in range(len(corners3_vals)):
        for j in range(len(corners3_vals[0])):
            if corners3_vals[i, j] > 0.1 * corners3_vals.max():
                corners3_coords.append([i, j])

    for corner in corners3_coords:
        x, y = corner
        # cv2.circle(img, (x, y), 1, [255, 255, 0], 0) # -1 means fill in the circle, 1 means thickness of circle

    plt.imshow(img)
    plt.suptitle("Shi Tomasi corners(yellow)")
    plt.show()

    corners = find_chessboard_corners(img)
    if corners is not None:
        to_txt_file("find_chess_board() corners ", corners)
        cropped_img, adjusted_corners = helper.crop_image(img, corners)
        # cv2.drawChessboardCorners(cropped_img, (7,7), adjusted_corners, True)
        for corner in adjusted_corners:
            x, y = corner.ravel()
            x, y = int(x), int(y)
            cv2.circle(cropped_img, (x, y), 1, [0, 255, 0], -1)
        plt.imshow(cropped_img)
        plt.suptitle("chessboard CORNERS")
        plt.show()
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("corners not found")
    canny_edge_detection(img)

# print("final real corners:", sorted_shi_tomasi_corners)
def group_into_squares(points):
    squares = {}
    for row in range(8):
        for col in range(8):
            # Each square is a set of four points
            top_left = tuple(points[row * 9 + col])
            top_right = tuple(points[row * 9 + col + 1])
            bottom_left = tuple(points[(row + 1) * 9 + col])
            bottom_right = tuple(points[(row + 1) * 9 + col + 1])
            # squares[(row, col)] = [top_left, top_right, bottom_left, bottom_right]
            squares[(top_left, top_right, bottom_left, bottom_right)] = (row, col)
    return squares

def chess_piece_string_to_char(piece):
    mapping = {
        'black_pawn': 'p',
        'black_rook': 'r',
        'black_knight': 'n',
        'black_bishop': 'b',
        'black_queen': 'q',
        'black_king': 'k',
        'white_pawn': 'P',
        'white_rook': 'R',
        'white_knight': 'N',
        'white_bishop': 'B',
        'white_queen': 'Q',
        'white_king': 'K',
    }
    return mapping.get(piece, None)  # Returns None if the piece is not found


corners_to_cell_indices_map = group_into_squares(sorted_shi_tomasi_corners)
print("corners_to_cell_indices_map", corners_to_cell_indices_map.keys())
board_with_pieces_img = cv2.imread("./IMG_5099.jpg")
board_with_pieces_img = cv2.resize(board_with_pieces_img, (0,0), fx=0.5 ** 3, fy=0.5 ** 3)
print("board_with_pieces_img.shape", board_with_pieces_img.shape)   
# classification_model = model.predict("./segmented-images-and-masks/PNGImages/IMG_5291.jpg", confidence=5, overlap=30).json()
classification_model = model.predict(board_with_pieces_img, confidence=5, overlap=30).json()

current_8x8_board = np.zeros((8, 8))
print("start 8x8 board", current_8x8_board)
for prediction in classification_model['predictions']:
    x, y = prediction['x'], prediction['y']
    print("x, y", x, y)
    cv2.circle(img, (int(x), int(y)), 1, [0, 255, 0], -1)
    for k, v in corners_to_cell_indices_map.items():
        if x > k[0][0] and x < k[1][0] and y > k[0][1] and y < k[2][1]:
            print("found piece", prediction['class'], "at", v)
            current_8x8_board[v[0], v[1]] = chess_piece_string_to_char(prediction['class'])
            break
print("final 8x8 board", current_8x8_board)
plt.imshow(img)
plt.show()
# trying something new:import cv2 as cv
# camera=cv2.VideoCapture(1)

# while True:
#     ret, frame_BGR_original=camera.read()
#     frame_BGR_resized = helper.resize_image(frame_BGR_original, 100)
#     frame_BGR_resized_2 = frame_BGR_resized

#     frame_GRAY = cv2.cvtColor(frame_BGR_resized, cv2.COLOR_BGR2GRAY)
#     frame_GRAY_blured=cv2.GaussianBlur(frame_GRAY,(5,5),0)

#     gray_corners = cv2.goodFeaturesToTrack(frame_GRAY, 100, 0.4, 5)
#     corners_array = np.int0(gray_corners)


#     #Display the corners found in the image
#     for i in corners_array:
#         x, y = i.ravel()
#         cv2.circle(frame_BGR_resized_2, (x, y), 3, [255, 255, 0], -1)

#     #frame_GRAY_cropped = my_functions.crop_image(frame_GRAY, corners_array)
#     frame_BGR_cropped= helper.crop_image(frame_BGR_resized,corners_array)


#     helper.open_in_location(frame_BGR_resized_2, "Shi Tomasi Corners", 00, 10)
#     helper.open_in_location(frame_BGR_cropped, "Original Frame Cropped", 800, 10)

#     key=cv2.waitKey(1) & 0xff
#     if key == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()
