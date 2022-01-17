#include "regions_tree.h"
#include <fstream>
#include <iostream>

RegionsTree::RegionsTree(std::string json_path)
{
    std::ifstream json_fl("../" + json_path);
    if (!json_fl)
        throw std::runtime_error("Failed to read in JSON region information from path: " + json_path);

    try
    {
        regions_json = json::parse(json_fl);
    }
    catch (json::parse_error& e)
    {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("JSON parse error was thrown");
    }
    json_fl.close();
}

int RegionsTree::GetParentRegionAtDepth(int region_idx, int depth) const
{
    // Get index of entry corresponding to region_idx in JSON.
    int idx_in_json = -1;
    for (int i = 0; i < regions_json.size(); ++i)
    {
        if (regions_json[i]["id"] == region_idx)
        {
            idx_in_json = i;
            break;
        }
    }
    if (idx_in_json == -1)
        throw std::runtime_error("Coulding find region with index " + std::to_string(region_idx) + " in JSON.");
    return  regions_json[idx_in_json]["structure_id_path"][depth];
}

int RegionsTree::GetIdOfRegionByName(std::string name, bool acronym)
{
    std::string label_str = acronym ? "acronym" : "name";
    for (auto& region : regions_json)
    {
        if (region[label_str] == name)
            return region["id"];
    }
    return 0;
}

std::string RegionsTree::GetNameOfRegionById(int id, bool acronym)
{
    std::string label_str = acronym ? "acronym" : "name";
    for (auto& region : regions_json)
    {
        if (region["id"] == id)
            return region[label_str];
    }
    return "";
}
