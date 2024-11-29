import random

class Game():
    def __init__(self):
        self._matrix = []
        self._score
    
    
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
        
        # Init score
        self._score = 0
    
    
    def game_log(self):
        board = ""
        for i in range(4):
            line = "|"
            for j in range(4):
                line += f"{self._matrix[i][j]}|"
            board += f"{line}\n"
        print(board)
                
    
    
    # Accessors
    @property
    def matrix(self):
        return self._matrix
    @property
    def score(self):
        return self._score
    