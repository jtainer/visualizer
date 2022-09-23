// 
// Complex number functions
//
// 2022, Jonathan Tainer
//

#include "complex.h"

Complex Complex::exp(unsigned int N) {
	Complex prod = { 1, 0 };

	for (unsigned int i = 0; i < N; i++) {
		Complex tmp;

		tmp.real = (prod.real * this->real) - (prod.imag * this->imag);
		tmp.imag = (prod.real * this->imag) + (prod.imag * this->real);

		prod = tmp;
	}

	return prod;
}
