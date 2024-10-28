#include "fileio.h"

void generate_input(std::string fname, int n){

    std::ofstream FILE(fname);
    std::vector<float> out;

    for(int i=0; i<n; ++i){
        float rng = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        FILE << rng << std::endl;
    }

    FILE.close();

}