#ifndef GAME_H
#define GAME_H

#include <vector>
#include <string>

class Game {
public:
    Game();
    Game(std::vector<std::vector<int>> weights);

    // Accessors
    const std::vector<std::vector<int>>& getGrid() const;
    int getScore() const;

    // State functions
    void init();
    void log() const;

    // Game functions
    void left();
    void right();
    void up();
    void down();

    // Game over check
    int isGameOver() const;

private:
    std::vector<std::vector<int>> _grid;
    int _score;
    std::vector<std::vector<int>> _weightDict;

    // Matrix manipulation functions
    void stack();
    void combine();
    void reverse();
    void transpose();
    bool checkZeroes() const;
    void addNewTile();

    // Helper functions for moves
    bool horizontalMoveExists() const;
    bool verticalMoveExists() const;

    void logCell(const std::string& bgColor, const std::string& value) const;
};

#endif // GAME_H