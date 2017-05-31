#include "pthread.h"

int main(int argc, char *argv[]) {
    pthread_mutex_t Mutex;

    pthread_mutexattr_t RecAttr;
    pthread_mutexattr_init(&RecAttr);
    pthread_mutexattr_settype(&RecAttr, PTHREAD_MUTEX_RECURSIVE);

    pthread_mutex_init(&Mutex, &RecAttr);

    // static pthread_mutex_t recmutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER;

    return 0;
}
