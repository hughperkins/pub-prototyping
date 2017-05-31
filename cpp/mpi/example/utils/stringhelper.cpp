#include <string>
#include <vector>
#include <sstream>
using namespace std;

#include "stringhelper.h"

vector<string> split(const string &str, const string &separator ) {
	vector<string> splitstring;
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

string trim( const string &target ) {

   int origlen = target.size();
   int startpos = -1;
   for( int i = 0; i < origlen; i++ ) {
      if( target[i] != ' ' && target[i] != '\r' && target[i] != '\n' ) {
         startpos = i;
         break;
      }
   }
   int endpos = -1;
   for( int i = origlen - 1; i >= 0; i-- ) {
      if( target[i] != ' ' && target[i] != '\r' && target[i] != '\n' ) {
         endpos = i;
         break;
      }      
   }
   if( startpos == -1 || endpos == -1 ) {
      return "";
   }
   return target.substr(startpos, endpos-startpos + 1 );
}

