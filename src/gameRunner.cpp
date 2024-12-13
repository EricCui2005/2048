#include "gameRunner.h"

std::mutex GameRunner::coutMutex;
std::mutex GameRunner::fileMutex;

void GameRunner::runTrials(int startRun, int numTrials) {
    MCTSPlayer player(100, 10, 100, 0.9);
    Game game;
    uint32_t wins = 0, losses = 0;

    std::vector<GameData> data;
    data.reserve(1000);

    std::string filepath = "../../data/run" + std::to_string(startRun) + ".csv";

    std::ofstream csvfile(filepath, std::ios::out | std::ios::binary);

    {
        std::lock_guard<std::mutex> lock(fileMutex);
        if(!csvfile) {
            std::cerr << "Failed to open: " << filepath << '\n';
            return;
        }
    }

    for(int i = 0; i < numTrials; i++) {
        game.init();
        data.clear();

        while(true) {
            const auto& currentState = game.getGrid();
            char move = player.move(&game)[0];

            data.push_back({move, currentState});

            switch(move) {
                case 'l': game.left(); break;
                case 'r': game.right(); break;
                case 'u': game.up(); break;
                case 'd': game.down(); break;
            }

            int gameState = game.isGameOver();
            if(gameState == 1) {
                wins++;
                {
                    std::lock_guard<std::mutex> lock(fileMutex);
                    for(const auto& entry : data) {
                        writeToFile(csvfile, entry);
                    }
                    csvfile.flush();
                }
                break;
            } else if(gameState == -1) {
                losses++;
                break;
            }
            
        }

        {
            std::lock_guard<std::mutex> lock(coutMutex);
            //std::cout << "Run " << startRun << " - W: " << wins << " L: " << losses << '\n';
        }
    }
}

void GameRunner::writeToFile(std::ofstream& file, const GameData& data) {
    std::string move = "";
    switch(data.move) {
        case 'l': move = "left"; break;
        case 'r': move = "right"; break;
        case 'u': move = "up"; break;
        case 'd': move = "down"; break;
    }
    file << move << ",\"[[";
    for(size_t i = 0; i < GRID_SIZE; i++) {
        for(size_t j = 0; j < GRID_SIZE; j++) {
            file << data.state[i][j];
            if(j < GRID_SIZE - 1) file << ", ";
        }
        if(i < GRID_SIZE - 1) file << "], [";
    }
    file << "]]\"\n";
}