// 
// Struct to store complex numbers
//
// 2022, Jonathan Tainer
//

#ifndef COMPLEX_H
#define COMPLEX_H

struct Complex {
	float real;
	float imag;

	Complex exp(unsigned int N);
};

#endif
