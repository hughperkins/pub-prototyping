#include <iostream>
#include <cstdlib>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
using namespace std;

#include "mpi.h"

string exec(string cmd) {
    std::string result = "";
try{
   cout << "cmd [" << cmd << "]" << endl;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    while(!feof(pipe)) {
        if(fgets(buffer, 128, pipe) != NULL)
                result += buffer;
    }
    pclose(pipe);
} catch(...) {
   cout << "caught exception " << endl;
}
    return result;
}

//int sigaction(int signum, const struct sigaction *act,
 //                    struct sigaction *oldact) {
  // cout << "sigaction ig " << signum << endl;
//}

void mysighandler(int sig ) {
   cout << "sighandler sig " << sig << endl;
}

void dumpfds() {
   cout << "rdonly " << O_RDONLY << " wronly " << O_WRONLY << " rdwr " << O_RDWR << endl;
   for( int i = 0; i < 30; i++ ) {
      int fd = fcntl(i, F_GETFD );
      int fl = fcntl(i, F_GETFL );
      int own = fcntl(i, F_GETOWN );
      int lease = fcntl(i, F_GETLEASE);
      cout << "fd " << i << " fl " << fl << " own " << own << " fd " << fd << " lease " << lease <<
        " sig " << fcntl(i, F_GETSIG ) << endl;
   }
}

int main(int argc, char *argv[] ) {
//   cout << stdin << endl;
   dup2( 0, 20 );
   dup2 (1, 21 );
   dup2 (2, 22 );

   MPI_Init( &argc, &argv );

   int rank, numprocs;
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   MPI_Comm_size( MPI_COMM_WORLD, &numprocs );

//close(20);
   dumpfds();

   int childpid = 0;
   if( ( childpid = fork() ) != 0 ) {
       cout << "parent process" << endl;
       sleep(1);
       cout << " parent about to call finalize" << endl;
       MPI_Finalize();
       cout << "parent finished" << endl;
   } else {
  //     cout << "child process begin" << endl;
//dumpfds();
   for( int i = 0; i < 20; i++ ) {
close(i);
}
//dumpfds();
dup2(20,0);
dup2(21,1);
dup2(22,2);
cout << "child saying hi?" <<endl;
//dumpfds();
  //     MPI_Init( &argc, &argv );
    //   cout << "child process end" << endl;
      execl("/data/dev/machinelearning/cpp/prototyping/build/mpichild","/data/dev/machinelearning/cpp/prototyping/build/mpichild");
   }

  // write( 6, "hello", 5 );
  // fsync(6);

/*
for( int i = 0; i < 60; i++ ) {
  // cout << "routing signal " << i << endl;
//   sigaction( i 
   signal( i, mysighandler );
}
signal( 13, SIG_IGN);
signal( 17, SIG_IGN);
string childresult = "";
try{
//system("bash -c ./mpichild");
cout << "calling child 1" << endl;
   dumpfds();
   childresult = exec("./mpichild");
   dumpfds();
cout << "calling child 2" << endl;
   childresult = exec("./mpichild");
cout << "calling child 3" << endl;
} catch( ... ) {
cout << "parent caught exception" << endl;
}
//MPI_Init(&argc, &argv );
   cout << "node " << rank << " " << childresult;

signal( 13, SIG_IGN);
signal( 17, SIG_IGN);
*/
   return 0;
}


