import random
import game.colors as c

class Game():
    def __init__(self):
        self._matrix = []
        self._score = 0
    
    
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
        board = ""
        for i in range(4):
            line = "|"
            for j in range(4):
                line += f"{c.ANSI_COLORS[self._matrix[i][j]]}{self._matrix[i][j]}\033[0m|"
            board += f"{line}\n"
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
                    self._score += self._matrix[i][j]
    
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
            return 1
        elif not self.check_zeroes() and not self.horizontal_move_exists() and not self.vertical_move_exists():
            return -1
        else:
            return 0
            
            
    
    

    
    
    
    