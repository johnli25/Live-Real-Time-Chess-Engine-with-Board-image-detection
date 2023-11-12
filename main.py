import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import stockfish
import chess
import chess.svg
import chess.engine

engine_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"

def returnChessboard8x8(board_img_nparray):
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        board = chess.Board()
        print(board)

        info = engine.analyse(board, chess.engine.Limit(time=1.0))

        recommened_move = info["pv"][0]
        return recommened_move

print(returnChessboard8x8(0))
    

