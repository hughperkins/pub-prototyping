mkdir ./main
cat > main/hello-world.cc <<'EOF'
#include "lib/hello-greet.h"
#include "main/hello-time.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string who = "world";
  if (argc > 1) {
    who = argv[1];
  }
  std::cout << get_greet(who) <<std::endl;
  print_localtime();
  return 0;
}
EOF
cat > main/hello-time.h <<'EOF'
#ifndef MAIN_HELLO_TIME_H_
#define MAIN_HELLO_TIME_H_

void print_localtime();

#endif
EOF
cat > main/hello-time.cc <<'EOF'
#include "main/hello-time.h"
#include <ctime>
#include <iostream>

void print_localtime() {
  std::time_t result = std::time(nullptr);
  std::cout << std::asctime(std::localtime(&result));
}
EOF
mkdir ./lib
cat > lib/hello-greet.h <<'EOF'
#ifndef LIB_HELLO_GREET_H_
#define LIB_HELLO_GREET_H_

#include <string>

std::string get_greet(const std::string &thing);

#endif
EOF
cat > lib/hello-greet.cc <<'EOF'
#include "lib/hello-greet.h"
#include <string>

std::string get_greet(const std::string& who) {
  return "Hello " + who;
}
EOF
