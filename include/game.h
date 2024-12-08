#ifndef GAME_H
#define GAME_H

#include "colors.h"
#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>

class Game {
public:
    static constexpr size_t GRID_SIZE = 4;
    using Grid = std::array<std::array<uint16_t, GRID_SIZE>, GRID_SIZE>;

    Game();
    explicit Game(const Grid& weights);

    // Accessors
    const Grid& getGrid() const noexcept { return _grid; }
    uint32_t getScore() const noexcept { return _score;}

    // State functions
    void init() noexcept;
    void log() const;

    // Game functions
    void left() noexcept;
    void right() noexcept;
    void up() noexcept;
    void down() noexcept;

    // Game over check
    int isGameOver() const noexcept;

private:
    Grid _grid;
    Grid _weights;
    uint32_t _score;

    // Matrix manipulation functions
    void stackLeft() noexcept;
    void combineLeft() noexcept;
    void reverse() noexcept;
    void transpose() noexcept;
    bool hasEmptyCell() const noexcept;
    void addNewTile() noexcept;
    bool canMergeHorizontal() const noexcept;
    bool canMergeVertical() const noexcept;
};

#endif // GAME_H