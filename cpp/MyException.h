#pragma once

#include <string>
using namespace std;

class MyException : public exception { 
public:
   string message;
   MyException( string message );
   virtual ~MyException() throw();
   virtual const char* what() const throw();
};


