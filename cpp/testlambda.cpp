#include <iostream>
#include <functional>

void callFunc(std::function<void()> &fn) {
    std::cout << "calling fn... " << &fn << std::endl;
    std::cout << "calling fn..." << std::endl;
    fn();
    std::cout << "... called fn" << std::endl;
}

void test1() {
    std::function<void()>myfunc = []{
        std::cout << "from anon lambda :-)" << std::endl;
    };
    std::cout << &myfunc << std::endl;
    std::cout << &myfunc << std::endl;
    std::function<void()>myfunc2 = []{
        std::cout << "another lambda :-)" << std::endl;
    };    
    std::cout << &myfunc << std::endl;
    std::cout << &myfunc2 << std::endl;

    callFunc(myfunc);
    callFunc(myfunc2);
    void *pf1 = &myfunc;
    void *pf2 = &myfunc2;
    (*(std::function<void()> *)pf1)();
    (*(std::function<void()> *)pf2)();
}

void test2() {
    int i = 3;
    std::function<void()>myfunc = [i]{
        std::cout << "from anon lambda :-). i was " << i << std::endl;
    };
    std::cout << &myfunc << std::endl;
    std::cout << &myfunc << std::endl;
    std::cout << (long)&myfunc << std::endl;
    // std::cout << (long)myfunc << std::endl;
    std::cout << " from cast: " << *(long *)(char *)(&myfunc) << std::endl;
    i = 7;
    std::function<void()>myfunc2 = [i]{
        std::cout << "another lambda :-) i was " << i << std::endl;
    };    
    std::cout << &myfunc << std::endl;
    std::cout << &myfunc2 << std::endl;
    std::cout << " from cast: " << *(long *)(char *)(&myfunc) << std::endl;
    std::cout << " from cast: " << *(long *)(char *)(&myfunc2) << std::endl;

    callFunc(myfunc);
    callFunc(myfunc2);
    void *pf1 = &myfunc;
    void *pf2 = &myfunc2;
    (*(std::function<void()> *)pf1)();
    (*(std::function<void()> *)pf2)();

    std::function<void()>myfunccopy = myfunc;
    myfunccopy();
    callFunc(myfunccopy);
    std::cout << " from cast: " << *(long *)(char *)(&myfunccopy) << std::endl;
    std::cout << " add func " << (long)&myfunc << " " << (long)&myfunccopy << std::endl;

    myfunccopy = myfunc2;
    std::cout << "aftre reassign:" << std::endl;
    myfunccopy();
    std::cout << " from cast: " << *(long *)(char *)(&myfunccopy) << std::endl;
    std::cout << " add func " << (long)&myfunc << " " << (long)&myfunccopy << std::endl;

    myfunccopy = myfunc;
    std::cout << "aftre re-reassign:" << std::endl;
    myfunccopy();
    std::cout << " from cast: " << *(long *)(char *)(&myfunccopy) << std::endl;
    std::cout << " add func " << (long)&myfunc << " " << (long)&myfunccopy << std::endl;

    std::function<void()>f1 = myfunc;
    std::function<void()>f2 = myfunc2;
    std::cout << std::endl;


    std::cout << "f1: " << *(long *)(char *)&f1 << std::endl;
    std::cout << "f2: " << *(long *)(char *)&f2 << std::endl;

    std::function<void()> f1copy = f1;
    std::cout << "\nafter copy f1 into f1copy:" << std::endl;
    std::cout << "addresses of f1 and f1copy differ: " << &f1 << " " << &f1copy << std::endl;
    std::cout << "but underlying pointers identical: " <<
        *(long *)(char *)&f1 << " " << *(long *)(char *)(&f1copy) << std::endl;

    std::cout << "\n after assign f2 to f1copy" << std::endl;
    f1copy = f2;
    std::cout << "now the underlying pointer of f1copy matches f2's:" << std::endl;
    std::cout << "but underlying pointers identical: " <<
        *(long *)(char *)&f2 << " " << *(long *)(char *)&f1copy << std::endl;
}

int main(int argc, char *argv[]) {
    test2();
    return 0;
}
