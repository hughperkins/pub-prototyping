#include <iostream>

#include "yaml-cpp/yaml.h"

using namespace std;

int main(int argc, char *argv[]) {
    YAML::Node myyaml;
    cout << "isnull " << myyaml.IsNull() << endl;
    myyaml = YAML::LoadFile("../test_yaml.yaml");
    cout << "isnull " << myyaml.IsNull() << endl;
    cout << myyaml << endl;
    // cout << "foo" << endl;
    for(auto it=myyaml.begin(); it != myyaml.end(); it++) {
        cout << it->first.as<std::string>() << endl;
        // cout << it->second.as<std::string>() << endl;
    }
    cout << "foox? " << myyaml["foox"] << endl;
    if(myyaml["foox"]) {
        cout << "found foox" << endl;
    }
    if(myyaml["foo"]) {
        cout << "found foo" << endl;
    }
    cout << myyaml["foo"].as<std::string>() << endl;
    cout << myyaml["someint"].as<int32_t>() << endl;
    cout << myyaml["blah"]["1"].as<std::string>() << endl;
    cout << myyaml["blah"][1].as<std::string>() << endl;
    cout << myyaml["blah"]["3"].as<std::string>() << endl;
    cout << myyaml["paris"][0].as<std::string>() << endl;
    cout << myyaml["paris"][1].as<std::string>() << endl;

    cout << myyaml["somebool"].as<bool>() << endl;
    cout << myyaml["anotherbool"].as<bool>() << endl;
    if(myyaml["doesntexist"]) {
        cout << myyaml["doesntexist"].as<bool>() << endl;
    }
    return 0;
}
