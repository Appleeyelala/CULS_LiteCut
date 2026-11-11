// CPU-only regression of the actual replay solver and AIGER writer.
#include "algorithms/rewrite.h"
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int left[9] = {}, right[9] = {};
    const int edges[5][2] = {{0, 2}, {1, 4}, {2, 4}, {8, 10}, {12, 6}};
    const auto managerLiteral = [](int literal) { return literal < 2 ? literal ^ 1 : literal; };
    for (int i = 0; i < 5; ++i) {
        left[i + 4] = managerLiteral(edges[i][0]);
        right[i + 4] = managerLiteral(edges[i][1]);
    }
    int outputs[5] = {1, 0, 8, 10, 16};  // AIGMan's constant convention.
    rw::CPUSolver solver;
    for (int repetition = 0; repetition < 2; ++repetition) {
        solver.Reset(3, 5, 8, left, right, outputs);
        solver.WriteAig((std::string(argv[1]) + "_" + std::to_string(repetition) + ".aig").c_str());
    }
    return 0;
}
