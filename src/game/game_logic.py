import random
import game.colors as c

NORMAL = [[1, 1, 1, 1], 
          [1, 1, 1, 1], 
          [1, 1, 1, 1], 
          [1, 1, 1, 1]]

HEURISTIC_A = [[1, 2, 3, 4], 
               [2, 3, 4, 5], 
               [3, 4, 5, 6], 
               [4, 5, 6, 7]]

HEURISTIC_B = [[0, 1, 2, 3], 
               [7, 6, 5, 4], 
               [8, 9, 10, 11], 
               [15, 14, 13, 12]]

HILL = [[1, 1, 1, 1], 
        [1, 2, 2, 1], 
        [1, 2, 2, 1], 
        [1, 1, 1, 1]]


class Game():
    def __init__(self):
        self._matrix = []
        self._score = 0
        
        # Reward dict for heuristic MCTS
        self._weight_dict = NORMAL
    
    
    """Accessors"""
    @property
    def matrix(self):
        return self._matrix
    @property
    def score(self):
        return self._score
    
    
    """State Functions"""
    
    def game_init(self):
        
        # Create matrix of zeroes
        self._matrix = [[0] * 4 for _ in range(4)]
        
        # Fill 2 random cells with 2s
        row = random.randint(0, 3)
        col = random.randint(0, 3)
        self._matrix[row][col] = 2
        while(self._matrix[row][col] != 0):
            row = random.randint(0, 3)
            col = random.randint(0, 3)
        self._matrix[row][col] = 2
    
    
    def game_log(self):
        cell_width = 7  # Width of each cell
        cell_height = 3  # Height of each cell
        border_horizontal = "-" * (cell_width * 4 + 5)  # Top/bottom border of the board

        def format_cell(value):
            """Format the cell with a white foreground and ANSI-colored background."""
            bg_color = c.ANSI_COLORS[value]  # Assume this is the background color code
            value_str = f"{value}" if value != 0 else " "
            padded_value = value_str.center(cell_width - 2)  # Center the value in the cell
            # Return the entire cell with background color, including padding
            return f"{bg_color} {padded_value} \033[0m"

        board = border_horizontal + "\n"
        for row in self._matrix:
            # Create a "row block" for each row of the board
            row_lines = [""] * cell_height
            for value in row:
                bg_color = c.ANSI_COLORS[value]  # Background color for the entire cell
                formatted_cell = format_cell(value)
                # Add formatted cell to each line of the block
                for line_idx in range(cell_height):
                    if line_idx == cell_height // 2:  # Center row with the number
                        row_lines[line_idx] += f"|{formatted_cell}"
                    else:  # Empty rows above and below the number use the same background color
                        row_lines[line_idx] += f"|{bg_color} {' ' * (cell_width - 2)} \033[0m"
            # Complete each row block
            for line in row_lines:
                board += line + "|\n"
            board += border_horizontal + "\n"
        print(board)
                
    
    """Matrix Manipulation Functions"""
    
    def stack(self):
        new_matrix = [[0] * 4 for _ in range(4)]
        
        # Iterating over all cells and stacking them to the left
        for i in range(4):
            fill_position = 0
            for j in range(4):
                if self._matrix[i][j] != 0:
                    new_matrix[i][fill_position] = self._matrix[i][j]
                    fill_position += 1
        self._matrix = new_matrix
    
    
    def combine(self):
        for i in range(4):
            for j in range(3):
                if self._matrix[i][j] != 0 and self._matrix[i][j] == self._matrix[i][j + 1]:
                    self._matrix[i][j] *= 2
                    self._matrix[i][j + 1] = 0
                    self._score += (self._matrix[i][j] * self._weight_dict[i][j])
    
    # Flips a matrix horizontally 
    def reverse(self):
        new_matrix = []
        for i in range(4):
            new_matrix.append([])
            for j in range(4):
                new_matrix[i].append(self._matrix[i][3 - j])
        self._matrix = new_matrix
    
    # Swapping rows and columns
    def transpose(self):
        new_matrix = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                new_matrix[i][j] = self._matrix[j][i]
        self._matrix = new_matrix
    
    
    # Checkinf if there are any zeroes in the grid
    def check_zeroes(self):
        if any(0 in row for row in self._matrix):
            return True
        return False
    
    # Add a new 2 or 4 tile randomly to an empty cell
    def add_new_tile(self):
        row = random.randint(0, 3)
        col = random.randint(0, 3)
        while(self._matrix[row][col] != 0):
            row = random.randint(0, 3)
            col = random.randint(0, 3)
        self._matrix[row][col] = random.choice([2, 4])
    
    
    """Arrow-Press Functions"""
        
    def left(self):
        self.stack()
        self.combine()
        self.stack()
        if self.check_zeroes():
            self.add_new_tile()
    
    
    def right(self):
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        if self.check_zeroes():
            self.add_new_tile()
    
    
    def up(self):
        self.transpose()
        self.stack()
        self.combine()
        self.stack()
        self.transpose()
        if self.check_zeroes():
            self.add_new_tile()
        
    
    def down(self):
        self.transpose()
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.transpose()
        if self.check_zeroes():
            self.add_new_tile()
    
    
    """Functions to check for possible moves"""
    
    def check_zeroes(self):
        if any(0 in row for row in self._matrix):
            return True
        return False
    
    def horizontal_move_exists(self):
        for i in range(4):
            for j in range(3):
                if self._matrix[i][j] == self._matrix[i][j + 1]:
                    return True
        return False
    
    
    def vertical_move_exists(self):
        for i in range(3):
            for j in range(4):
                if self._matrix[i][j] == self._matrix[i + 1][j]:
                    return True
        return False
    

    """Game Over"""

    def game_over(self):
        if any(2048 in row for row in self.matrix):
            return 1 # Win
        elif not self.check_zeroes() and not self.horizontal_move_exists() and not self.vertical_move_exists():
            return -1 # loss
        else:
            return 0
