#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

vector<string> split(const string& line, char delimiter) {
    vector<string> tokens;
    string token;
    stringstream ss(line);

    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

double safe_stod(const string& s) {
    try {
        if (s.empty()) return 0.0;
        return stod(s);
    } catch (...) {
        return 0.0;
    }
}

int main() {
    string input_file = "data/orderbook.csv";
    string output_file = "results/cpp_features.csv";

    ifstream input(input_file);
    ofstream output(output_file);

    if (!input.is_open()) {
        cerr << "Could not open " << input_file << endl;
        return 1;
    }

    output << "timestamp,symbol,mid,spread,bid_qty,ask_qty,top5_bid,top5_ask,"
           << "imbalance,microprice_dev,trade_side,trade_qty,"
           << "relative_spread,depth_ratio,queue_pressure,log_trade_qty\n";

    string line;
    getline(input, line); // skip header

    long long count = 0;

    while (getline(input, line)) {
        vector<string> row = split(line, ',');

        if (row.size() < 14) continue;

        string timestamp = row[0];
        string symbol = row[1];

        double mid = safe_stod(row[2]);
        double spread = safe_stod(row[3]);
        double bid_qty = safe_stod(row[6]);
        double ask_qty = safe_stod(row[7]);
        double top5_bid = safe_stod(row[8]);
        double top5_ask = safe_stod(row[9]);
        double imbalance = safe_stod(row[10]);
        double microprice_dev = safe_stod(row[11]);
        int trade_side = static_cast<int>(safe_stod(row[12]));
        double trade_qty = safe_stod(row[13]);

        double relative_spread = spread / (mid + 1e-9);
        double depth_ratio = top5_bid / (top5_ask + 1e-9);
        double queue_pressure = bid_qty / (ask_qty + 1e-9);
        double log_trade_qty = log1p(trade_qty);

        output << timestamp << ","
               << symbol << ","
               << mid << ","
               << spread << ","
               << bid_qty << ","
               << ask_qty << ","
               << top5_bid << ","
               << top5_ask << ","
               << imbalance << ","
               << microprice_dev << ","
               << trade_side << ","
               << trade_qty << ","
               << relative_spread << ","
               << depth_ratio << ","
               << queue_pressure << ","
               << log_trade_qty << "\n";

        count++;

        if (count % 100000 == 0) {
            cout << "Processed " << count << " rows" << endl;
        }
    }

    cout << "Done. Processed " << count << " rows." << endl;
    cout << "Saved to " << output_file << endl;

    return 0;
}