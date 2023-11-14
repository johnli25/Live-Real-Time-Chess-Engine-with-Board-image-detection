import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import stockfish
import chess
import chess.svg
import chess.engine

engine_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"

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
    print(piece)
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
    