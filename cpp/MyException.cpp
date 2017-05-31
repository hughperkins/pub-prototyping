#include "MyException.h"

MyException::MyException( string message ) {
    this->message = message;
}
MyException::~MyException() throw() {
}
const char* MyException::what() const throw()
{
  return message.c_str();
}

