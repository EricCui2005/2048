#ifndef GAME_RUNNER_H
#define GAME_RUNNER_H

#include "game.h"
#include "mcts.h"
#include <array>
#include <thread>
#include <mutex>
#include <fstream>
#include <iostream>
#include <string>

class GameRunner {
public:
    static constexpr size_t GRID_SIZE = 4;
    using Grid = std::array<std::array<uint16_t, GRID_SIZE>, GRID_SIZE>;

    struct GameData {
        char move;
        Grid state;
    };

    static void runTrials(int startRun, int numTrials);

private:
    static std::mutex coutMutex;
    static std::mutex fileMutex;
    static void writeToFile(std::ofstream& file, const GameData& data);
};

#endif // GAME_RUNNER_H