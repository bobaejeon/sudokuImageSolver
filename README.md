# Sudoku Image Solver
Solves sudoku given as an image using ***OpenCV Python*** and ***Python Tessaract OCR***

### Roughly summarized steps
1. Preprocess the image includes sudoku board (hint: use perspective transform and make it binary) 
2. Divide the board into 9*9=81 cells
3. Extract numbers from each cell (**OCR**) and store them
4. Solve the sudoku using backtracking algorithm
5. Show the result on the given sudoku board (as you can see below)

#### Things I can improve
<ul>
  Considering a better way to preprocess the given image.<br>
  Implementing deep learning method to extract numbers.
 </ul>
 
 ### Result
 <img width="792" alt="sudokuSolver" src="https://user-images.githubusercontent.com/67196344/103080939-3bb8e880-461a-11eb-8d62-b1eb6e2b8571.PNG">
