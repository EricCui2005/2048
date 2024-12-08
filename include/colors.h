#ifndef COLORS_H
#define COLORS_H

#include <unordered_map>
#include <string>

// Terminal Colors
const std::unordered_map<int, std::string> ANSI_COLORS = {
    {0, "\033[48;2;0;0;0m"},
    {2, "\033[48;2;252;239;230m"},
    {4, "\033[48;2;242;232;203m"},
    {8, "\033[48;2;245;182;130m"},
    {16, "\033[48;2;242;148;70m"},
    {32, "\033[48;2;255;119;92m"},
    {64, "\033[48;2;230;76;46m"},
    {128, "\033[48;2;237;226;145m"},
    {256, "\033[48;2;252;225;48m"},
    {512, "\033[48;2;255;219;74m"},
    {1024, "\033[48;2;240;185;34m"},
    {2048, "\033[48;2;250;215;77m"}
};

// GUI Colors
const std::string GRID_COLOR = "#a39489";
const std::string EMPTY_CELL_COLOR = "#c2b3a9";
const std::string SCORE_LABEL_FONT = "Verdana, 24"; // Font info represented as a string
const std::string SCORE_FONT = "Helvetica, 36, bold";
const std::string GAME_OVER_FONT = "Helvetica, 48, bold";
const std::string GAME_OVER_FONT_COLOR = "#ffffff";
const std::string WINNER_BG = "#ffcc00";
const std::string LOSER_BG = "#a39489";

const std::unordered_map<int, std::string> CELL_COLORS = {
    {2, "#fcefe6"},
    {4, "#f2e8cb"},
    {8, "#f5b682"},
    {16, "#f29446"},
    {32, "#ff775c"},
    {64, "#e64c2e"},
    {128, "#ede291"},
    {256, "#fce130"},
    {512, "#ffdb4a"},
    {1024, "#f0b922"},
    {2048, "#fad74d"}
};

const std::unordered_map<int, std::string> CELL_NUMBER_COLORS = {
    {2, "#695c57"},
    {4, "#695c57"},
    {8, "#ffffff"},
    {16, "#ffffff"},
    {32, "#ffffff"},
    {64, "#ffffff"},
    {128, "#ffffff"},
    {256, "#ffffff"},
    {512, "#ffffff"},
    {1024, "#ffffff"},
    {2048, "#ffffff"}
};

// Font sizes are stored as strings for now, but could be parsed or structured further
const std::unordered_map<int, std::string> CELL_NUMBER_FONTS = {
    {2, "Helvetica, 55, bold"},
    {4, "Helvetica, 55, bold"},
    {8, "Helvetica, 55, bold"},
    {16, "Helvetica, 50, bold"},
    {32, "Helvetica, 50, bold"},
    {64, "Helvetica, 50, bold"},
    {128, "Helvetica, 45, bold"},
    {256, "Helvetica, 45, bold"},
    {512, "Helvetica, 45, bold"},
    {1024, "Helvetica, 40, bold"},
    {2048, "Helvetica, 40, bold"}
};

#endif // COLORS_H
