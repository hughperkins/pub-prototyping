#include <iostream>
using namespace std;

#include "json/json.h" 
using namespace Json;

int main( int argc, char *argv[] ) {
    CharReader reader;
    Value value;
    string errors;
    string doc = "{ 'name': 'paris', 'age': 18 }";
    bool result = reader.parse( &doc.c_str()[0], &doc.c_str()[doc.length()-1], &value, &errors );
    cout << value.isObject() << endl;
    cout << value["name"] << endl;
    cout << "result: " << result << endl;
    cout << errors << endl;
    return 0;
}

