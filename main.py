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
current_8x8_board = returnBoardConfigurationFromImage("test.png")
print(returnNextBestMove(current_8x8_board))

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

def canny_edge_detection(img):
    # convert img to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200, apertureSize=5) # apertureSize is the size of Sobel kernel (3,5,7) used for find image gradients, 100 = min threshold for hysterisis, 200 = max threshold for hysterisis procedure

    # find contours: contours returns a list of contours, hierarchy describes the child-parent relationships between contours (e.g. if one contour is inside another contour)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # sort countours by descending area: internally, it temporarily convert array of points to enclosed area and determine closed area using cv2.contourArea
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    chessboard_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # extract perimeter of contour

        # approximate the contour to a polygon by simplifying contour shape to another shape with fewer vertices
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) # approx is a list of points representing the polygon
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
        print("chessboard not found")


output_directory = [jpg_directory + "IMG_5262.jpg", "./online_walnut_chess.jpg"]
for img_path in output_directory:
    img = cv2.imread(img_path)

    # downscale image:
    img = cv2.resize(img, (0,0), fx=0.5 ** 2, fy=0.5 ** 2)

    corners = find_chessboard_corners(img)
    print("corners", corners)
    if corners is not None:
        cv2.drawChessboardCorners(img, (7,7), corners, True)
        plt.imshow(img)
        plt.suptitle("chessboard CORNERS")
        plt.show()
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("corners not found")

    canny_edge_detection(img)


