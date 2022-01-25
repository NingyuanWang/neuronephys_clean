#include "circadian.h"
using namespace std;

std::vector<float> convertStringVectortoFloatVector(const std::vector<std::string>& stringVector) {
    std::vector<float> floatVector(stringVector.size());
    std::transform(stringVector.begin(), stringVector.end(), floatVector.begin(), [](const std::string& val)
        {
            return stof(val);
        });
    return floatVector;
}

vector<vector<float>> Circadian::readCSV(const string& filename)
{
    vector< vector<float> > result;
    string csvLine;
    fstream file(filename, ios::in);
    if (!file.is_open())
    {
        cout << "File not found!\n";
        system("pause");
    }

    // read every line from the stream
    while (getline(file, csvLine))
    {
        istringstream csvStream(csvLine);
        vector<string> csvColumn;
        string csvElement;
        // read every element from the line that is seperated by commas
        // and put it into the vector or strings
        while (getline(csvStream, csvElement, ','))
        {
            csvColumn.push_back(csvElement);
        }
        result.push_back(convertStringVectortoFloatVector(csvColumn));
    }

    return result;
}

std::vector<float> Circadian::get_params_at_time(float ctime) const //ctime is in 24hrs
{
    std::vector<float> result(2);
    std::vector<float> lower_vec;
    std::vector<float> higher_vec;
    float a;

    const float dt = 0.00586081; //the circadian table is interpolated equidistantly with this dt
    float ctime_hr = std::fmod(ctime, 24);
    int lower_ind = floor(ctime_hr / dt);
    vector< vector<float> >::const_iterator ind_ptr = circtable.begin();
    advance(ind_ptr, lower_ind);
    for (vector<float>::const_iterator j = ind_ptr->begin(); j != ind_ptr->end(); j++) {
        a = *j;
        lower_vec.push_back(a);

    }

    vector< vector<float> >::const_iterator up_ptr = circtable.begin();
    advance(up_ptr, lower_ind + 1);
  

    for (vector<float>::const_iterator j = up_ptr->begin(); j != up_ptr->end(); j++) {
        a = *j;
        higher_vec.push_back(a);
    }
  
    for (int k = 0; k < 2; k++)
        result[k] = lower_vec[k + 1] + (higher_vec[k + 1] - lower_vec[k + 1]) * (ctime_hr / dt - lower_ind);

  
    return result;

}