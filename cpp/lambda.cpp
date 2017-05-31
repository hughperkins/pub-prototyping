#include <iostream>
#include <set>
#include <functional>
using namespace std;

// void walk(myfunc *fn) {
//     fn("hello");
// }

void doSomething(std::function<void(string)> fn) {
    fn("susan");
    fn("june");
}

int main(int argc, char *argv[]) {
    set<string> seen;
    auto fn = [&seen](string name) {
        cout << "hi! " << name << endl;
        seen.insert(name);
    };
    fn("john");
    fn("peter");
    doSomething(fn);
    for(auto it=seen.begin(); it != seen.end(); it++) {
        cout << "in seen: " << *it << endl;
    }
    // set<string> seen;
    // walk()
    return 0;
}
