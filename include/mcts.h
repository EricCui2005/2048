#ifndef MCTS_H
#define MCTS_H

#include "game.h"
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

class MCTS {

public:
    MCTS(Game* game, int simulations, int rollouts, double c, double discount);
    std::string move();

private:
    struct MoveData {
        int N;
        double Q;
    };

    std::unordered_map<std::string, MoveData> _moves;
    int _totalVisits;
    Game* _game;
    int _simulations;
    int _rollouts;
    double _c;
    double _discount;

    double div(double dividend, double divisor);
    std::string choice();
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