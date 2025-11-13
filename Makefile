clean:
	rm -rf ./matmul ./mat_transpose ./softmax ./copy_compute_test
matmul:
	nvcc -o ./matmul ./matmul.cu
mat_transpose:
	nvcc -o ./mat_transpose ./mat_transpose.cu
softmax:
	nvcc -o ./softmax ./softmax.cu
copy_compute_test:
	nvcc -o ./copy_compute_test ./copy_compute_test.cu
retrieval_kernel:
	nvcc -o ./retrieval_kernel ./retrieval_kernel.cu
top_kernel:
	nvcc -o ./top_kernel ./top_kernel.cu
