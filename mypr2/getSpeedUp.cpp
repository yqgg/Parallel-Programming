#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main() {
    // Open the input text file
    //change file name according to which size and which fill factor speed up wanted
    ifstream inputFile("./size3/out-.10.txt"); 

    // Check if the input file is open
    if (!inputFile.is_open()) {
        cerr << "Error opening input file." << endl;
        return 1;
    }

    // Open the output text file
    ofstream outputFile("output_speedup.txt");

    // Check if the output file is open
    if (!outputFile.is_open()) {
        cerr << "Error opening output file." << endl;
        inputFile.close();
        return 1;
    }

    string line;
    while (getline(inputFile, line)) {
        // Check if the line contains "Speed up: "
        size_t found = line.find("Speed up: ");
        if (found != string::npos) {
            // Extract the numeric value after "Speed up: "
            string speedUpValue = line.substr(found + 10); // 10 is the length of "Speed up: "
            
            // Convert the string to a double and write to the output file
            outputFile << stod(speedUpValue) << endl;
        }
    }

    // Close the files
    inputFile.close();
    outputFile.close();

    return 0;
}
