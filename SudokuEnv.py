import numpy as np

class SudokuEnv:
    def __init__(self, puzzle=None):
        self.board = np.zeros((9, 9), dtype=int)
        self.original = None
        self.done = False
        self.max_steps = 1000
        self.current_step = 0

        if puzzle is not None:
            self.load_puzzle(puzzle)

    def load_puzzle(self, puzzle):
        self.board = np.array(puzzle, dtype=int)
        if self.board.shape != (9, 9):
            raise ValueError(f"Puzzle debe tener forma (9, 9), pero tiene {self.board.shape}")
        self.original = self.board.copy()


    def reset(self, puzzle=None):
        self.current_step = 0
        self.done = False

        if puzzle is not None:
            self.load_puzzle(puzzle)
        elif self.original is None or not isinstance(self.original, np.ndarray) or self.original.shape != (9, 9):
            # Si no hay puzzle válido, inicializa a tablero vacío
            self.board = np.zeros((9, 9), dtype=int)
            self.original = self.board.copy()
        else:
            # Restaurar el puzzle original
            self.board = self.original.copy()
        
        return self.get_state()

    
    def get_state(self):
        # Return the current board state
        return self.board.copy()
    
    def get_valid_actions(self):
        """Return a list of all valid actions in format (row, col, num)"""
        valid_actions = []
        
        for row in range(9):
            for col in range(9):
                if self.original[row][col] == 0 and self.board[row][col] == 0:
                    for num in range(1, 10):
                        if self.is_valid_move(row, col, num):
                            valid_actions.append((row, col, num))
        
        return valid_actions
    
    def get_action_mask(self):
        """Create a binary mask of valid actions (1=valid, 0=invalid)"""
        mask = np.zeros(729, dtype=int)
        
        for row in range(9):
            for col in range(9):
                if self.original[row][col] == 0 and self.board[row][col] == 0:
                    for num in range(1, 10):
                        action_idx = row * 81 + col * 9 + (num - 1)
                        mask[action_idx] = 1 if self.is_valid_move(row, col, num) else 0
        
        return mask
    
    def get_number_constraints(self):
        """Get information about which numbers are already used in each row/column/box"""
        rows = np.zeros((9, 9), dtype=int)  # rows[i][j] = 1 if number j+1 is in row i
        cols = np.zeros((9, 9), dtype=int)  # cols[i][j] = 1 if number j+1 is in col i
        boxes = np.zeros((9, 9), dtype=int)  # boxes[i][j] = 1 if number j+1 is in box i
        
        for r in range(9):
            for c in range(9):
                if self.board[r][c] != 0:
                    num = self.board[r][c] - 1  # Convert to 0-based index
                    rows[r][num] = 1
                    cols[c][num] = 1
                    box_idx = (r // 3) * 3 + (c // 3)
                    boxes[box_idx][num] = 1
        
        return rows, cols, boxes

    def is_valid_move(self, row, col, num):
        # Check if the cell is part of the original puzzle
        if self.original[row][col] != 0:
            return False
        
        # Check if the number is valid
        if num > 9 or num < 1:
            return False
        
        # Check row constraint
        if num in self.board[row, :]:
            return False
        
        # Check column constraint
        if num in self.board[:, col]:
            return False
        
        # Check 3x3 box constraint
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if num in self.board[start_row:start_row + 3, start_col:start_col + 3]:
            return False
        
        return True

    def step(self, action):
        row, col, num = action
        
        # Initialize reward with a small step penalty
        reward = -1
        info = {'valid_move': False}

        if self.done:
            return self.get_state(), -10, True, info
        
        # Check if move is valid
        if self.is_valid_move(row, col, num):
            self.board[row][col] = num
            reward = 10
            info['valid_move'] = True
            
            # Additional rewards for completing constraints
            row_complete = np.all(np.sort(self.board[row, :]) == np.arange(1, 10))
            col_complete = np.all(np.sort(self.board[:, col]) == np.arange(1, 10))
            
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            box = self.board[box_row:box_row+3, box_col:box_col+3].flatten()
            box_complete = np.all(np.sort(box) == np.arange(1, 10))
            
            if row_complete:
                reward += 50
                info['row_complete'] = True
            if col_complete:
                reward += 50
                info['col_complete'] = True
            if box_complete:
                reward += 50
                info['box_complete'] = True
        else:
            # Higher penalty for invalid moves
            reward = -10
        
        self.current_step += 1
        
        # Check if puzzle is solved
        self.done = self.check_done()
        if self.done:
            reward += 500  # Big reward for solving the puzzle
            info['solved'] = True
        
        # Check if we exceeded max steps
        if self.current_step >= self.max_steps:
            info['max_steps_exceeded'] = True
            self.done = True
        
        return self.get_state(), reward, self.done, info

    def check_done(self):
        """Check if the puzzle is solved correctly"""
        if not np.all(self.board > 0):  # Check if all cells are filled
            return False
            
        # Check all rows
        for i in range(9):
            if set(self.board[i, :]) != set(range(1, 10)):
                return False
                
        # Check all columns
        for i in range(9):
            if set(self.board[:, i]) != set(range(1, 10)):
                return False
                
        # Check all 3x3 boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = self.board[box_row:box_row+3, box_col:box_col+3].flatten()
                if set(box) != set(range(1, 10)):
                    return False
                    
        return True
    
    def render(self):
        """Print the current state of the board"""
        print("\n+-------+-------+-------+")
        for i in range(9):
            print("|", end=" ")
            for j in range(9):
                print(self.board[i, j] if self.board[i, j] != 0 else ".", end=" ")
                if j % 3 == 2:
                    print("|", end=" ")
            print()
            if i % 3 == 2:
                print("+-------+-------+-------+")
        print()

