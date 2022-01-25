#include <vector> 
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class Circadian
{
private:
	vector<vector<float>> circtable;
	static vector<vector<float>> readCSV(const string& filename);


public:
	
	vector<float> get_params_at_time(float ctime) const;
	Circadian(const std::string& filename) {
		circtable = readCSV(filename);
	}

};
