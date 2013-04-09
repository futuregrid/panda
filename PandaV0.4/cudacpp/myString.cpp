#include <cudacpp/myString.h>

namespace cudacpp
{
	String::String(){}  
	String::String(const int n){}

	String::String(const int n, const char fill){}
	String::String(const char * const str){}
	String::String(const String & rhs){}
	String::~String(){}

	String & String::operator = (const String & rhs){
		String str;
		return str;
	}
	String   String::operator + (const String & rhs) const{
		String str;
		return str;
	}
	String  String::operator + (const char * const rhs) const{
		String str;
		return str;
	}
	String & String::operator += (const String & rhs){
		String str;
		return str;
	}
	String & String::operator += (const char * const rhs){
		String str;
		return str;
	}

	int String::size() const{
		return 0;
	}
	String String::substr(const int start, const int len ) const
	{
		String str;
		return str;
	}
	const char * String::c_str() const
	{
		return 0;
	}

}


      
