#include "game.h"

Game::Game() : _score(0) {
    _weights.fill({1, 1, 1, 1});
}

Game::Game(const Grid& weights) : _weights(weights), _score(0) {}

void Game::init() {
    _grid.fill({0, 0, 0, 0});
    _score = 0;
    addNewTile();
    addNewTile();
}

void Game::log() const {
    static constexpr int CELL_WIDTH = 7;
    static constexpr int CELL_HEIGHT = 3;
    static const std::string borderHorizontal(CELL_WIDTH * GRID_SIZE + 5, '-');

    for(const auto& row: _grid) {
        std::cout << borderHorizontal << '\n';
        for(int h = 0; h < CELL_HEIGHT; ++h) {
            for(const auto& value : row) {
                std::string bgColor = ANSI_COLORS.count(value) ? ANSI_COLORS.at(value) : ANSI_COLORS.at(0);
                if(h == CELL_HEIGHT / 2) {
                    std::string numStr = value ? std::to_string(value) : " ";
                    int padding_left = (CELL_WIDTH - 1 - numStr.length()) / 2;
                    int padding_right = CELL_WIDTH - 1 - padding_left - numStr.length();
                    
                    std::cout << '|' << bgColor 
                        << std::string(padding_left, ' ')
                        << numStr
                        << std::string(padding_right, ' ')
                        << "\033[0m";
                } else {
                    std::cout << '|' << bgColor
                        << std::string(CELL_WIDTH - 1, ' ')
                        << "\033[0m";
                }
            }
            std::cout << "|\n";
        }
    }
    std::cout << borderHorizontal << '\n';
}

void Game::left() noexcept {
    stackLeft();
    combineLeft();
    stackLeft();
    if (hasEmptyCell()) addNewTile();
}

void Game::right() noexcept {
    reverse();
    stackLeft();
    combineLeft();
    stackLeft();
    reverse();
    if (hasEmptyCell()) addNewTile();
}

void Game::up() noexcept {
    transpose();
    stackLeft();
    combineLeft();
    stackLeft();
    transpose();
    if (hasEmptyCell()) addNewTile();
}

void Game::down() noexcept {
    transpose();
    reverse();
    stackLeft();
    combineLeft();
    stackLeft();
    reverse();
    transpose();
    if (hasEmptyCell()) addNewTile();
}

void Game::stackLeft() noexcept {
    for (auto& row : _grid) {
        size_t writePos = 0;
        for (size_t readPos = 0; readPos < GRID_SIZE; ++readPos) {
            if (row[readPos] != 0) {
                if (writePos != readPos) {
                    row[writePos] = row[readPos];
                    row[readPos] = 0;
                }
                ++writePos;
            }
        }
    }
}

void Game::combineLeft() noexcept {
    for (auto& row : _grid) {
        for (size_t i = 0; i < GRID_SIZE-1; ++i) {
            if (row[i] != 0 && row[i] == row[i+1]) {
                row[i] *= 2;
                row[i+1] = 0;
                _score += row[i];
            }
        }
    }
}

void Game::reverse() noexcept {
    for (auto& row : _grid) {
        for (size_t i = 0; i < GRID_SIZE/2; ++i) {
            std::swap(row[i], row[GRID_SIZE-1-i]);
        }
    }
}

void Game::transpose() noexcept {
    for (size_t i = 0; i < GRID_SIZE; ++i) {
        for (size_t j = i+1; j < GRID_SIZE; ++j) {
            std::swap(_grid[i][j], _grid[j][i]);
        }
    }
}

void Game::addNewTile() noexcept {
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_int_distribution<> pos(0, GRID_SIZE-1);
    static thread_local std::uniform_int_distribution<> val(0, 1);

    int row, col;
    do {
        row = pos(gen);
        col = pos(gen);
    } while (_grid[row][col] != 0);

    _grid[row][col] = (val(gen) ? 2 : 4);
}

int Game::isGameOver() const noexcept {
    for (const auto& row : _grid) {
        for (auto val : row) {
            if (val == 2048) return 1;
        }
    }
    
    if (hasEmptyCell() || canMergeHorizontal() || canMergeVertical()) {
        return 0;
    }
    
    return -1;
}

bool Game::hasEmptyCell() const noexcept {
    for (const auto& row : _grid) {
        for (auto val : row) {
            if (val == 0) return true;
        }
    }
    return false;
}

bool Game::canMergeHorizontal() const noexcept {
    for (const auto& row : _grid) {
        for (size_t j = 0; j < GRID_SIZE-1; ++j) {
            if (row[j] == row[j+1] && row[j] != 0) return true;
        }
    }
    return false;
}

bool Game::canMergeVertical() const noexcept {
    for (size_t j = 0; j < GRID_SIZE; ++j) {
        for (size_t i = 0; i < GRID_SIZE-1; ++i) {
            if (_grid[i][j] == _grid[i+1][j] && _grid[i][j] != 0) return true;
        }
    }
    return false;
}
