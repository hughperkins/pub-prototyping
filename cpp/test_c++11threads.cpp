#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <stdexcept>


void run_from_thread(int tid) {
    std::cout << "run from thread, tid=" << tid << " actual id=" << std::this_thread::get_id() << std::endl;
}

struct Counter {
    Counter() {
    }
    void child() {
        std::lock_guard<std::recursive_mutex> guard(mu);
    }
    int increment() {
        // mu.lock();
        std::lock_guard< std::recursive_mutex > guard(mu);
        // child();
        value++;
        // mu.unlock();
        return value;
    }
    int operator()() {
        return value;
    }
    int value = 0;
    std::recursive_mutex mu;
};

class MyVars {
public:
    MyVars() {
        std::cout << "MyVars()" << std::endl;
    }
    ~MyVars() {
        std::cout << "~MyVars()" << std::endl;
    }
    Counter counter;
};

std::once_flag initedMyVars;
static MyVars *myVars = 0;
static void initMyVars() {
    myVars = new MyVars();
    std::cout << "inited myvars" << std::endl;
}
static MyVars *getMyVars() {
    std::call_once(initedMyVars, initMyVars);
    return myVars;
}

void test1() {
    // based on https://baptiste-wicht.com/posts/2012/03/cp11-concurrency-tutorial-part-2-protect-shared-data.html
    std::thread t1(run_from_thread, 0);
    t1.join();

    std::vector<std::thread> threads;
    for(int i=0; i < 5; i++) {
        threads.push_back(std::thread(run_from_thread, i));
    }
    for(int i=0; i < 5; i++) {
        threads[i].join();
    }

    threads.clear();

    const int numThreads = 500;
    // Counter counter;
    for(int i = 0; i < numThreads; i++) {
        threads.push_back(std::thread([](){
            MyVars *myVars = getMyVars();
            // std::cout << "thread using lambda, id=" << std::this_thread::get_id() << std::endl;
            for(int i = 0; i < 100; i++) {
                myVars->counter.increment();
            }
            // std::cout << "counter at end: " << counter() << std::endl;
        }));
    }
    for(int i=0; i < numThreads; i++) {
        threads[i].join();
    }

    std::cout << "counter at end: " << getMyVars()->counter() << std::endl;
}

thread_local MyVars *threadVars = nullptr;
std::mutex print_mutex;

void test_threadlocal() {
    std::vector< std::thread > threads;
    const int numThreads = 50;
    for(int i = 0; i < numThreads; i++) {
        threads.push_back(std::thread([i]() {
            int numCreations = 0;
            for(int i = 0; i < 10000; i++) {
                if(threadVars == 0) {
                    threadVars = new MyVars();
                    numCreations++;
                }
                threadVars->counter.increment();
            }
            std::lock_guard< std::mutex > guard(print_mutex);
            std::cout << "thread " << i << " counter " << threadVars->counter() << " vars " << (long)threadVars
                << " numCreations=" << numCreations << std::endl;
        }));
    }
    for(int i = 0; i < numThreads; i++) {
        threads[i].join();
    }
    std::cout << "end of test_threadlocal()" << std::endl;
}

int main(int argc, char *argv[]) {
    // test1();
    test_threadlocal();

    std::cout << "finished main" << std::endl;
    return 0;
}
