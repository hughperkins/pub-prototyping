// based on https://meetingcpp.com/tl_files/mcpp/slides/12/FunctionalProgrammingInC++11.pdf

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

template<int N>
struct Fac {
    static const int value = N * Fac<N - 1>::value;
};

template<>
struct Fac<0> {
    static const int value = 1;
};

// int fac(const int N) {
//     return Fac<N>::value;
// }

int main(int argc, char *argv[]) {
    std::vector<std::string> str{"some", "strings", "Paris", "go"};
    std::vector<int> lengths;
    std::transform(str.begin(), str.end(), back_inserter(lengths),
        [] (std::string s) { return s.length(); });
    for(int v : lengths) {
        std::cout << "v: " << v << std::endl;
    }
    auto new_end = std::remove_if(str.begin(), str.end(),
        [] (std::string s) { return (isupper(s[0])); });
    for(auto it=str.begin(); it != new_end; it++) {
        std::cout << "after filter: " << *it << std::endl;
    }
    std::string res = std::accumulate(str.begin(), str.end(),
        std::string(""), [](std::string a, std::string b) {
            return a + ":" + b;
        });
    std::cout << "std::accmumulate res: " << res << std::endl;

    std::cout << "Fac<5>::value = " << Fac<5>::value << std::endl;
    // std::cout << "fac(5) = " << fac(5) << std::endl;

    int a = 6;
    std::cout << Fac<a>::value << std::endl;

    return 0;
}
