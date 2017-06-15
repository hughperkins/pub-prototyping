#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>

// based on http://www.cprogramming.com/tutorial/functors-function-objects-in-c++.html
class MyFunctorClass {
public:
     MyFunctorClass(int x) : _x(x) {}
     int operator() (int y) { return _x + y; }
private:
    int _x;
 };

class Message {
public:
    const std::string getHeader(std::string &header_name) const {
        return headers.at(header_name);
    }
    std::map<std::string, std::string> headers;
};

class MessageSorter {
public:
    MessageSorter(const std::string &field) : _field(field) {}
    bool operator() (const Message &lhs, const Message &rhs) {
        return lhs.getHeader(_field) < rhs.getHeader(_field);
    }
    std::string _field;
};

void sortMessages() {
    std::vector<Message> messages;
    MessageSorter comparator("destination");
    sort(messages.begin(), messages.end(), comparator);
}

struct Abs {
    float operator()(float f) {
        std::cout << "Abs::operator()(float f)" << std::endl;
        return f > 0 ? f : -f;
    }
};

class Print {
public:
    void operator()(int elem) {
        std::cout << elem << " ";
    }
};

template<typename Iterator, typename Function>
void my_for_each(Iterator it, Iterator end, Function function) {
    for(;it != end; it++) {
        function(*it);
    }
}

template<int val>
void SetVal(int &elem) {
    elem = val;
};

class Add {
public:
    Add(int val) : val(val) {}
    void operator()(int &elem) {
        elem += val;
    }
private:
    int val;
};

// from http://www.bogotobogo.com/cplusplus/functors.php
void test2() {
    Abs abs;
    std::cout << "abs(-2.3f) = " << abs(-2.3f) << std::endl;

    std::vector<int> v = {1, 7, 9, 11};
    Print printit;
    std::for_each(v.begin(), v.end(), printit);
    std::cout << std::endl;

    my_for_each(v.begin(), v.end(), SetVal<6>);
    my_for_each(v.begin(), v.end(), printit);
    std::cout << std::endl;

    my_for_each(v.begin(), v.end(), Add(4));
    my_for_each(v.begin(), v.end(), printit);
    std::cout << std::endl;

    std::transform(v.begin(), v.end(), v.begin(),
        std::negate<int>());
    my_for_each(v.begin(), v.end(), printit);
    std::cout << std::endl;

    std::transform(v.begin(), v.end(),
        v.begin(),
        v.begin(),
        std::multiplies<int>());
    my_for_each(v.begin(), v.end(), printit);
    std::cout << std::endl;
}

template<typename T>
class Less {
public:
    bool operator()(const T &one, const T &two) {
        return one < two;
    }
};

template<typename T>
class Greater {
public:
    bool operator()(const T&one, const T& two) {
        return one > two;
    }
};

template<class Operation, typename T>
class MyBinder2nd {
public:
    MyBinder2nd(T second) : second(second) {}
    T operator()(T first) {
        return Operation(first, second);
    }
private:
    const T second;
};

// template<typename T>

void testBinder2nd() {
    // Less<int> comp;
    std::vector<int> v = {1, 7, 4, 9};
    // std::cout << comp(1, 3) << std::endl;
    std::sort(v.begin(), v.end(), Less<int>());
    for_each(v.begin(), v.end(), [](int val){
        std::cout << val << " "; });
    std::cout << std::endl;

    std::sort(v.begin(), v.end(), Greater<int>());
    for_each(v.begin(), v.end(), [](int val){
        std::cout << val << " "; });
    std::cout << std::endl;

    // MyBinder2nd<Less, int> less3(3);

    // MyBinder2nd<pow>()
    // std::cout << std::pow(2,3) << std::endl;

    // MyBinder2nd<pow> pow2(2.0f);

}

template<typename T>
class myrange {
public:
    myrange(T last) : last(last) {
        this->cur = 0;
    }
    const myrange &begin() {
        return *this;
    }
    const myrange &end() {
        return *this;
    }
    T operator*() {
        return cur;
    }
    void operator++() {
        cur++;
    }
    bool operator!=(const myrange &second) {
        return cur != last;
    }

    // T operator()() {
    //     std::cout << "myrange::operator()" << std::endl;
    //     return cur++;
    // }
private:
    T cur;
    T last;
};

void testGenerator() {
    // from https://en.wikipedia.org/wiki/Generator_(computer_programming)
    for(int i : myrange<int>(10)) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    // MyFunctorClass addFive(5);
    // std::cout << addFive(6) << std::endl;

    // test2();
    // testBinder2nd();
    testGenerator();

    return 0;
}
