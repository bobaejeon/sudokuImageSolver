import cv2 as cv
import numpy as np
import pytesseract
from random import shuffle
from imutils import contours
from skimage.segmentation import clear_border


# Preprocess given image
def preprocess():
    global src
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.GaussianBlur(src_gray, (5, 5), 0)  # to remove noise
    src_gray = cv.adaptiveThreshold(src_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 2)  # for binarization to make the board clear

    edge = cv.Canny(src_gray, 150, 250)  # to find edges
    contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)  # sort contours from the largest area(desc. order)

    #cv.drawContours(src, contours, 0, (0, 255, 0), 3)

    for i in range(len(contours)):
        approx = cv.approxPolyDP(contours[i], cv.arcLength(contours[i], True) * 0.02, True)
        # if the polygon has 4 vertices, that can be considered as a rectangle
        if len(approx) == 4:
            break  # first one must be the largest

    approx = approx.reshape(len(approx), np.size(approx[0]))

    xSubY = np.subtract(approx[:, 0], approx[:, 1])
    xAddY = approx.sum(axis=1)

    src_pts = np.zeros((4, 2), dtype=np.float32)
    src_pts[0, :] = approx[np.where(xAddY == np.min(xAddY))].reshape(2)  # min(x+y)
    src_pts[1, :] = approx[np.where(xSubY == np.max(xSubY))].reshape(2)  # max(x-y)
    src_pts[2, :] = approx[np.where(xAddY == np.max(xAddY))].reshape(2)  # max(x+y)
    src_pts[3, :] = approx[np.where(xSubY == np.min(xSubY))].reshape(2)  # min(x-y)

    return perspective_transform(src_pts, src_gray)


# Extract the game board from given image
def perspective_transform(src_pts, src_gray):
    global src
    w = int(max(abs(src_pts[1][0] - src_pts[0][0]), abs(src_pts[2][0] - src_pts[3][0])))
    h = int(max(abs(src_pts[1][1] - src_pts[2][1]), abs(src_pts[0][1] - src_pts[3][1])))

    dst_pts = np.array([[0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]]).astype(np.float32)

    pers_mat = cv.getPerspectiveTransform(src_pts, dst_pts)
    # game board to put result in
    editted = cv.warpPerspective(src, pers_mat, (w, h))
    # game board to get digit to solve
    preprocessed = cv.warpPerspective(src_gray, pers_mat, (w, h))

    return preprocessed, editted


# Divide the grid into 9*9=81 cells
def divide_grid(preprocessed):
    global board
    cnts = cv.findContours(preprocessed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    grid = preprocessed.copy()
    for c in cnts:
        area = cv.contourArea(c)
        if area < 800:
            cv.drawContours(grid, [c], -1, 0, -1)

    # get vertical, horizontal lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
    grid = cv.morphologyEx(grid, cv.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
    grid = cv.morphologyEx(grid, cv.MORPH_CLOSE, horizontal_kernel, iterations=4)

    # sort cells from top-bottom & left-right and store them in an array
    grid = 255 - grid
    cnts = cv.findContours(grid, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    sudoku_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv.contourArea(c)
        if area < 50000:
            row.append(c)
            if i % 9 == 0:
                (cnts, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(cnts)
                row = []

    # Extract numbers from each cell and store them in a sudoku board
    for i in range(9):
        for j in range(9):
            (x, y, w, h) = cv.boundingRect(sudoku_rows[i][j])
            cell = preprocessed[y:y + h, x:x + w]
            board[i][j] = find_number(cell)

    return sudoku_rows


# Extract number from a cell
def find_number(cell):
    cell = clear_border(cell)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    cell = cv.morphologyEx(cell, cv.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv.findContours(cell, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    num = 0

    if len(contours) != 0:
        contour = max(contours, key=cv.contourArea)

        if cv.contourArea(contour) > 100:
            text = pytesseract.image_to_string(cell, lang="eng", config='--psm 6 --oem 3')

            if '1' <= list(text)[0] <= '9':
                num = int(list(text)[0])
           
    return num


# See if the number is possible
def possible(y, x, num):
    global board
    for i in range(9):
        if board[i][x] == num:
            return False
    for i in range(9):
        if board[y][i] == num:
            return False
    col = x - x % 3
    row = y - y % 3
    for i in range(3):
        for j in range(3):
            if board[row + i][col + j] == num:
                return False
    return True


# Solve the puzzle
def solve():
    global board, solution
    numbers = np.arange(1, 10)
    for y in range(9):
        for x in range(9):
            if board[y][x] == 0:
                shuffle(list(numbers))
                for num in numbers:
                    if possible(y, x, num):
                        board[y][x] = num  # solved
                        solve()  # look for another empty element(recursive)
                        board[y][x] = 0  # if an empty element is not solvable, make the "already solved" states empty
                return  # no number is possible
    solution = board.copy()


# Show result in game board
def show_result(locations, editted):
    global solution

    for i in range(9):
        for j in range(9):
            (x, y, w, h) = cv.boundingRect(locations[i][j])
            cv.putText(editted, str(solution[i][j]), (x+10, y+35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv.imshow('solution', editted)


src = cv.imread("sudoku.jpg")

if src is None:
    print('Image load failed')
    exit()

#cv.imshow('src', src)

board = np.zeros((9, 9), dtype=int)  # initialize the game board
solution = np.zeros((9, 9), dtype=int) # store solution

preprocessed, edited = preprocess() # preprocessed image
locations = divide_grid(preprocessed) # location of each cells
solve()
cv.imshow('game', src)
show_result(locations, edited)

cv.waitKey()
cv.destroyAllWindows()
