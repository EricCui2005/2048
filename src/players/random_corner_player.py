import random

class RandomCorner:
    def move(self, game):
        return random.choice(['up', 'left', 'down'])
    