#ifndef MCTS_H
#define MCTS_H

#include "game.h"
#include <cmath>
#include <string>
#include <array>

class MCTS {

public:
    MCTS(Game* game, int simulations, int rollouts, double c, double discount);
    std::string move();

private:
    struct MoveData {
        int N = 0;
        double Q = 0.0;
    };

    static constexpr size_t NUM_MOVES = 4;
    static constexpr std::array<const char*, NUM_MOVES> MOVE_STRINGS = {"left", "right", "up", "down"};
    std::array<MoveData, NUM_MOVES> _moves;

    int _totalVisits;
    const int _simulations;
    const int _rollouts;
    const double _c;
    const double _discount;

    Game* _game;

    int choice();
};

class MCTSPlayer {

public:
    MCTSPlayer(int simulations, int rollouts, double c, double discount);
    std::string move(Game* game);

private:
    int _simulations;
    int _rollouts;
    double _c;
    double _discount;
};

#endif // MCTS_H