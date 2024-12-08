#include "game.h"
#include "colors.h"
#include <iostream>
#include <iomanip>
#include <random>

Game::Game() {
    _grid = std::vector<std::vector<int>>(4, std::vector<int>(4, 0));
    _score = 0;
    _weightDict = {
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1}
    };
}

Game::Game(std::vector<std::vector<int>> weights) {
    _grid = std::vector<std::vector<int>>(4, std::vector<int>(4, 0));
    _score = 0;
    _weightDict = weights;
}

const std::vector<std::vector<int>>& Game::getGrid() const {
    return _grid;
}

int Game::getScore() const {
    return _score;
}

void Game::init() {
    _grid = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    _score = 0;
    addNewTile();
    addNewTile();
}

void Game::log() const {
    const int cellWidth = 7;
    const int cellHeight = 3;
    const std::string borderHorizontal(cellWidth * 4 + 5, '-');

    auto formatCell = [&](int value) -> std::string {
        std::string bgColor = ANSI_COLORS.count(value) ? ANSI_COLORS.at(value) : ANSI_COLORS.at(0);
        std::string valueStr = value != 0 ? std::to_string(value) : " ";
        
        // Center the value
        int padding = (cellWidth - 2 - valueStr.length()) / 2;
        std::string paddedValue = std::string(padding, ' ') + valueStr + 
                                 std::string(cellWidth - 2 - padding - valueStr.length(), ' ');
        
        return bgColor + " " + paddedValue + " \033[0m";
    };

    std::string board = borderHorizontal + "\n";
    
    for (const auto& row : _grid) {
        std::vector<std::string> rowLines(cellHeight);
        
        for (int value : row) {
            std::string bgColor = ANSI_COLORS.count(value) ? ANSI_COLORS.at(value) : ANSI_COLORS.at(0);
            std::string formattedCell = formatCell(value);
            
            for (int lineIdx = 0; lineIdx < cellHeight; ++lineIdx) {
                if (lineIdx == cellHeight / 2) {
                    rowLines[lineIdx] += "|" + formattedCell;
                } else {
                    rowLines[lineIdx] += "|" + bgColor + " " + 
                                       std::string(cellWidth - 2, ' ') + 
                                       " \033[0m";
                }
            }
        }
        
        for (const auto& line : rowLines) {
            board += line + "|\n";
        }
        board += borderHorizontal + "\n";
    }
    
    std::cout << board;
}

void Game::left() {
    stack();
    combine();
    stack();
    if (checkZeroes()) addNewTile();
}

void Game::right() {
    reverse();
    stack();
    combine();
    stack();
    reverse();
    if (checkZeroes()) addNewTile();
}

void Game::up() {
    transpose();
    stack();
    combine();
    stack();
    transpose();
    if (checkZeroes()) addNewTile();
}

void Game::down() {
    transpose();
    reverse();
    stack();
    combine();
    stack();
    reverse();
    transpose();
    if (checkZeroes()) addNewTile();
}

int Game::isGameOver() const {
    if (std::any_of(_grid.begin(), _grid.end(),
                    [](const std::vector<int>& row) {
                        return std::find(row.begin(), row.end(), 2048) != row.end();
                    })) {
        return 1; // Win
    }
    if (!checkZeroes() && !horizontalMoveExists() && !verticalMoveExists()) {
        return -1; // Loss
    }
    return 0;
}


void Game::stack() {
    for (auto& row : _grid) {
        std::vector<int> newRow(4, 0);
        int fillPosition = 0;
        for (const auto& value : row) {
            if (value != 0) {
                newRow[fillPosition++] = value;
            }
        }
        row = newRow;
    }
}

void Game::combine() {
    for (auto& row : _grid) {
        for (size_t j = 0; j < 3; ++j) {
            if (row[j] != 0 && row[j] == row[j + 1]) {
                row[j] *= 2;
                row[j + 1] = 0;
                _score += row[j];
            }
        }
    }
}

void Game::reverse() {
    for (auto& row : _grid) {
        std::reverse(row.begin(), row.end());
    }
}

void Game::transpose() {
    std::vector<std::vector<int>> newMatrix(4, std::vector<int>(4, 0));
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            newMatrix[i][j] = _grid[j][i];
        }
    }
    _grid = newMatrix;
}

bool Game::checkZeroes() const {
    for (const auto& row : _grid) {
        if (std::find(row.begin(), row.end(), 0) != row.end()) {
            return true;
        }
    }
    return false;
}

void Game::addNewTile() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 3);

    int row, col;
    do {
        row = dist(gen);
        col = dist(gen);
    } while (_grid[row][col] != 0);

    _grid[row][col] = (dist(gen) % 2 == 0) ? 2 : 4;
}

bool Game::horizontalMoveExists() const {
    for (const auto& row : _grid) {
        for (size_t j = 0; j < 3; ++j) {
            if (row[j] == row[j + 1]) return true;
        }
    }
    return false;
}

bool Game::verticalMoveExists() const {
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (_grid[i][j] == _grid[i + 1][j]) return true;
        }
    }
    return false;
}