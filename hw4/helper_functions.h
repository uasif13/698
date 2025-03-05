#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS

int dot_product_cuda(int my_work, int my_rank, int*vecC, int*vecD);
int sum(int size, int* data)
{
	int accumulate;
	for (int i = 0; i < size; i++) {
		accumulate += data[i];
	}
	return accumulate;
}

#endif /* HELPER_FUNCTIONS */
