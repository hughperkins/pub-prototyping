#pragma once

#include <vector>
#include <string>
#include <sstream>

template<typename T>
std::string toString(T val ) { // not terribly efficient, but works...
   std::ostringstream myostringstream;
   myostringstream << val;
   return myostringstream.str();
}

std::vector<std::string> split(const std::string &str, const std::string &separator = " " );
std::string trim( const std::string &target );
/*
class ExtendedString : public string {
public:
   ExtendedString( const string& origstring ) {
      *this = origstring;
   }
   ExtendedString( const char * origstring ) {
      *this = origstring;
   }
   vector<ExtendedString> split( string separator ) const {
      vector<ExtendedString> splitstring;
      ExtendedString str = *this;
      int start = 0;
      int npos = str.find(separator);
      while (npos != (int)str.npos ) {
         splitstring.push_back( str.substr(start, npos-start) );
         start = npos + 1;
         npos = str.find(separator, start);
      }
      splitstring.push_back( str.substr( start ) );
      return splitstring;
   }
   ExtendedString trim() const {
      return ::trim(*this);
   }
};
*/
inline double atof( std::string stringvalue ) {
   return atof(stringvalue.c_str());
}
inline double atoi( std::string stringvalue ) {
   return atoi(stringvalue.c_str());
}

// returns empty string if off the end of the number of available tokens
inline std::string getToken( std::string targetstring, int tokenIndexFromZero, std::string separator = " " ) {
   std::vector<std::string> splitstring = split( targetstring, separator );
   if( tokenIndexFromZero < splitstring.size() ) {
      return splitstring[tokenIndexFromZero];
   } else {
      return "";
   }
}

inline std::string toLower(std::string in ) {
   int len = in.size();
   char buffer[len + 1];
   for( int i = 0; i < len; i++ ) {
      char thischar = in[i];
      thischar = tolower(thischar);
      buffer[i] = thischar;
   }
   buffer[len] = 0;
   return std::string(buffer);
}
