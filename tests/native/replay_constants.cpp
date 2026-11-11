// CPU-only regression of the actual replay solver and AIGER writer.
#include "algorithms/rewrite.h"
#include <stdexcept>
#include <string>

static void check_dependency_and_alias_chains(rw::CPUSolver & solver) {
    solver.numInputs = 2;
    solver.n = 6;
    solver.fanin0[3] = 2;
    solver.fanin1[3] = 4;
    solver.fanin0[4] = solver.fanin0[5] = 1;
    solver.fanin1[4] = 7;  // complemented node 3
    solver.fanin1[5] = 9;  // complemented node 4
    solver.fanin0[6] = 6;  // dependency through fanin0, not an alias
    solver.fanin1[6] = 2;
    std::vector<int> marks(7, 0);
    if (!solver.ReplacementDependsOn(5, 3, marks, 1) ||
        !solver.ReplacementDependsOn(6, 3, marks, 2) ||
        solver.ReplacementDependsOn(3, 5, marks, 3) || solver.Fanin(11) != 7)
        throw std::runtime_error("transitive dependency or complement-parity regression");
    // Closing the three-node ring formerly escaped the direct-fanin check.
    solver.fanin0[3] = 1;
    solver.fanin1[3] = 10;
    bool rejected = false;
    try { solver.Fanin(6); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("alias cycle was not rejected");
    // A long valid chain must resolve without using the native call stack.
    solver.n = 100004;
    solver.fanin0[3] = 2;
    solver.fanin1[3] = 4;
    int parity = 1;
    for (int node = 4; node <= solver.n; ++node) {
        solver.fanin0[node] = 1;
        solver.fanin1[node] = 2 * (node - 1) + (node & 1);
        parity ^= node & 1;
    }
    if (solver.Fanin(2 * solver.n + 1) != (6 ^ parity))
        throw std::runtime_error("long valid alias chain changed literal polarity");
}

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
    check_dependency_and_alias_chains(solver);
    for (int repetition = 0; repetition < 2; ++repetition) {
        solver.Reset(3, 5, 8, left, right, outputs);
        solver.WriteAig((std::string(argv[1]) + "_" + std::to_string(repetition) + ".aig").c_str());
    }
    return 0;
}
