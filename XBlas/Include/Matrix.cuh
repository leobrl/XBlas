#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <exception>
#include <vector>
#include <string>
#include <type_traits>
#include "CudaManager.cuh"
#include "Architecture.hpp"
#include "MemoryLayout.cuh"
#include "MatrixColumn.cuh"
#include <cusolverDn.h>


#ifndef XBLAS_MATRIX
#define XBLAS_MATRIX

namespace XBlas
{
	template<typename T>
	class Matrix;

	template<class T>
	using __matrix__ = std::shared_ptr<Matrix<T>>;

	template<class T>
	using __buffer__ = std::shared_ptr<MemoryBuffer<T>>;

	template<class T>
	using __column__ = std::shared_ptr<MatrixColumn<T>>;

	/* Implementation of column major matrix*/
	template<class T>
	class  __declspec(dllexport) Matrix : public MemoryLayout<T>
	{
		static_assert(sizeof(T) == 4, "Matrix only supports type with size of 32 bits.");
		static_assert(std::is_arithmetic<T>::value, "Matrix only supports arithmetic types.");

	private:

		size_t nRows_;
		size_t nColumns_;
		std::vector<__column__<T>> columns;

		Matrix(size_t nRows, size_t nColumns, Architecture arch);
		Matrix(const __buffer__<T> memoryBuffer, long nRows, long nColumns);

		void SetColumns();

	public:

		Matrix() = delete;
		Matrix(Matrix&) = delete;
		Matrix(Matrix&&) = delete;
		Matrix& operator=(Matrix const&) = delete;
		Matrix& operator=(Matrix const&&) = delete;

		static __matrix__<T> Build(long nRows, long nColumns, Architecture arch);

		static __matrix__<T> Build(const __buffer__<T> memoryBuffer, long nRows, long nColumns);

		static __matrix__<T> Diagonal(const std::vector<T> diag, Architecture arch);

		static __matrix__<T> Identity(long nRows, Architecture arch);

		void Move(Architecture arch) override;

		const virtual size_t Length() const
		{
			return buffer->GetCapacity();
		}

		const size_t nRows() const
		{
			return this->nRows_;
		}

		const size_t nColumns() const
		{
			return this->nColumns_;
		}

		__column__<T>& operator [] (size_t index);

		const __column__<T>& operator [] (size_t index) const;

		__matrix__<T> Transpose();

		template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
		std::shared_ptr<Matrix<U>>  operator* (const std::shared_ptr<Matrix<U>> B);

		template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
		std::shared_ptr<Vector<U>>  operator* (const std::shared_ptr<Vector<U>> V);

		template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
		std::shared_ptr<Matrix<U>> Inverse();

		template<class U>
		std::shared_ptr<Matrix<U>> Cast();

	private:
		__matrix__<T> CuTranspose();

		template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
		std::shared_ptr<Matrix<U>> CuInverse();
	
		//TODO replace with cuda version
		template<class U>
		void HostCast(U* const destination, const T* source)
		{
			for (int pos = 0; pos < nRows_*nColumns_; pos++)
			{
				destination[pos] = (U)source[pos];
			}
		}
	};

	template<class T>
	__matrix__<T> Matrix<T>::Build(long nRows, long nColumns, Architecture arch)
	{
		if (nRows*nColumns < 0)
			throw std::out_of_range("Marix dimensions must be positive numbers");

		return std::shared_ptr<Matrix<T>>(new Matrix<T>(nRows, nColumns, arch));
	}

	template<class T>
	Matrix<T>::Matrix(size_t nRows, size_t nColumns, Architecture arch)
		: nRows_{ nRows }, nColumns_{ nColumns }
	{
		size_t capacity = nRows*nColumns;
		SetBuffer(MemoryBuffer<T>::MemAlloc(capacity, arch));
		SetColumns();
	}

	template<class T>
	__matrix__<T> Matrix<T>::Build(const __buffer__<T> memoryBuffer, long nRows, long nColumns)
	{
		if (nRows*nColumns < 0)
			throw std::out_of_range("Marix dimensions must be positive numbers");

		return std::shared_ptr<Matrix<T>>(new Matrix<T>(memoryBuffer, nRows, nColumns));
	}

	template<class T>
	Matrix<T>::Matrix(const __buffer__<T> memoryBuffer, long nRows, long nColumns)
	{
		if (nRows*nColumns < 0)
			throw std::out_of_range("Marix dimensions must be positive numbers");

		nRows_ = nRows;
		nColumns_ = nColumns;

		SetBuffer(memoryBuffer);
		SetColumns();
	}

	template<class T>
	__matrix__<T> Matrix<T>::Diagonal(const std::vector<T> diag, Architecture arch)
	{
		// TODO : make a cuda version
		int nRows = diag.size();
		auto diagonalMatrix = std::shared_ptr<Matrix<T>>(new Matrix<T>(nRows, nRows, Host));
		//cudaMemset(identity->buffer->GetPtr(), 0, nRows*nRows * sizeof(T));
		for (int i = 0; i < nRows; ++i)
		{
			for (int j = 0; j < nRows; ++j)
			{
				if (i == j)
					(diagonalMatrix->operator[](i))->operator[](j) = diag[i];
				else
					(diagonalMatrix->operator[](i))->operator[](j) = 0.0;
			}
		}

		switch (arch)
		{
		case XBlas::Host:
			break;
		case XBlas::Device:
			diagonalMatrix->Move(Device);
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}

		return diagonalMatrix;
	}

	template<class T>
	__matrix__<T> Matrix<T>::Identity(long nRows, Architecture arch)
	{
		if (nRows < 0)
			throw std::out_of_range("Marix dimensions must be positive numbers");

		std::vector<T> diag(nRows);
		std::fill(diag.begin(), diag.end(), 1.0);

		return Diagonal(diag, arch);
	}

	template<class T>
	void Matrix<T>::Move(Architecture arch)
	{
		MemoryLayout<T>::Move(arch);

		T* colPtr{ buffer->GetPtr() };
		for (size_t col = 0; col < nColumns_; ++col)
		{
			columns[col]->Move(colPtr, arch);
			colPtr += nRows_;
		}
	}

	template<class T>
	std::shared_ptr<MatrixColumn<T>>& Matrix<T>::operator[](size_t index)
	{
		return columns[index];
	}

	template<class T>
	const std::shared_ptr<MatrixColumn<T>> & Matrix<T>::operator[](size_t index) const
	{
		return columns[index];
	}

	template<class T>
	void Matrix<T>::SetColumns()
	{
		T* colPtr{ buffer->GetPtr() };

		for (size_t col = 0; col < nColumns_; ++col)
		{
			columns.push_back(MatrixColumn<T>::Build(colPtr, nRows_, buffer->GetArchitecture()));
			colPtr += nRows_;
		}
	}

	template<class T>
	template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
	std::shared_ptr<Matrix<U>>  Matrix<T>::operator* (const std::shared_ptr<Matrix<U>> B)
	{
		//TODO refactor this
		std::shared_ptr<Matrix<U>> C = Matrix<U>::Build(this->nRows_, B->nColumns(), Architecture::Device);

		CudaManager& manager = CudaManager::GetInstance();
		cublasStatus_t status = cublasStatus_t::CUBLAS_STATUS_SUCCESS;
		int leadingDimensionB = B->nRows();
		int leadingDimensionC = C->nRows();
		int nColC = C->nColumns();
		float alpha = 1.0f;
		float beta = 0.0f;

		switch (buffer->GetArchitecture())
		{
		case Host:
			this->Move(Device);
			B->Move(Device);

			status = cublasSgemm(manager.GetCublasHandle(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				leadingDimensionB,
				nColC,
				leadingDimensionC,
				&alpha,
				(float*)this->buffer->GetPtr(),
				leadingDimensionB,
				(float*)B->buffer->GetPtr(),
				leadingDimensionC,
				&beta,
				(float*)C->buffer->GetPtr(),
				leadingDimensionB);
			checkCudaStatus(status);

			this->Move(Host);
			B->Move(Host);
			C->Move(Host);

			break;
		case Device:

			status = cublasSgemm(manager.GetCublasHandle(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				leadingDimensionB,
				nColC,
				leadingDimensionC,
				&alpha,
				(float*)B->buffer->GetPtr(),
				leadingDimensionB,
				(float*)C->buffer->GetPtr(),
				leadingDimensionC,
				&beta,
				(float*)this->buffer->GetPtr(),
				leadingDimensionB);
			checkCudaStatus(status);
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}

		return C;
	}

	template<class T>
	template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
	std::shared_ptr<Vector<U>>  Matrix<T>::operator* (const std::shared_ptr<Vector<U>> X)
	{
		std::shared_ptr<Vector<U>> Y = XBlas::Vector<U>::Build(nRows_, Device);

		CudaManager& manager = CudaManager::GetInstance();
		cublasStatus_t status = cublasStatus_t::CUBLAS_STATUS_SUCCESS;

		int m = nRows_;
		int n = nColumns_;
		float alpha = 1.0;
		int lda = nRows_;
		int incx = 1;
		float beta = 0.0;
		int incy = 1;

		switch (buffer->GetArchitecture())
		{
		case Host:
			this->Move(Device);
			X->Move(Device);

			status = cublasSgemv(manager.GetCublasHandle(),
				CUBLAS_OP_N,
				m,
				n,
				&alpha,
				(float*)buffer->GetPtr(),
				lda,
				(float*) X->buffer->GetPtr(),
				incx,
				&beta,
				(float*)Y->buffer->GetPtr(),
				incy);
			checkCudaStatus(status);

			this->Move(Host);
			X->Move(Host);
			Y->Move(Host);
			
			break;
		case Device:

			status = cublasSgemv(manager.GetCublasHandle(),
				CUBLAS_OP_N,
				m,
				n,
				&alpha,
				(float*)buffer->GetPtr(),
				lda,
				(float*)X->buffer->GetPtr(),
				incx,
				&beta,
				(float*)Y->buffer->GetPtr(),
				incy);

			checkCudaStatus(status);
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}
		return Y;
	}

	template<class T>
	__matrix__<T> Matrix<T>::Transpose()
	{
		std::shared_ptr<Matrix<T>> C;
		switch (buffer->GetArchitecture())
		{
		case Host:
		{
			this->Move(Device);

			C = CuTranspose();

			this->Move(Host);
			C->Move(Host);
		}
		break;
		case Device:
		{
			C = CuTranspose();
		}
		break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}

		return C;

	}

	template<class T>
	template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
	std::shared_ptr<Matrix<U>> Matrix<T>::Inverse()
	{
		std::shared_ptr<Matrix<U>> inverse;
		switch (buffer->GetArchitecture())
		{
		case Host:
		{
			this->Move(Device);
			inverse = CuInverse();
			this->Move(Host);
			inverse->Move(Host);
		}
		break;
		case Device:
			inverse = CuInverse();
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}
		return inverse;
	}

	template<class T>
	template<class U>
	std::shared_ptr<Matrix<U>> Matrix<T>::Cast()
	{
		std::shared_ptr<Matrix<U>> C = Matrix<U>::Build(nRows_, nColumns_, Host);
		switch (buffer->GetArchitecture())
		{
		case Host:
		{
			HostCast(C->buffer->GetPtr(), buffer->GetPtr());
		}
		break;
		case Device:
			this->Move(Host);
			HostCast(C->buffer->GetPtr(), buffer->GetPtr());
			this->Move(Device);
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}

		return C;
	}

	// Helper funtions on device

	template<class T>
	template<class U = T, class = std::enable_if<std::is_same<U, float>::value>::type>
	std::shared_ptr<Matrix<U>> Matrix<T>::CuInverse()
	{
		CudaManager& manager = CudaManager::GetInstance();
		cusolverStatus_t  status = cusolverStatus_t::CUSOLVER_STATUS_SUCCESS;
		std::shared_ptr<Matrix<U>> inverse = Matrix<U>::Identity(nRows_, Device);

		cudaStream_t stream = NULL;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		cusolverDnSetStream(manager.GetCusolverHandle(), stream);

		// Copy buffer because LU factorization writes over it
		std::shared_ptr<Matrix<U>> LU = Matrix<U>::Build(nRows_, nColumns_, Device);
		cudaMemcpy(LU->buffer->GetPtr(), buffer->GetPtr(), (buffer->GetCapacity()) * sizeof(U), cudaMemcpyDeviceToDevice);

		// Calculate buffer size for LU factorization algorithm
		int bufferSize = 0;
		status = cusolverDnSgetrf_bufferSize(manager.GetCusolverHandle(),
			nRows_,
			nRows_,
			(float*)LU->buffer->GetPtr(),
			nRows_,
			&bufferSize);
		cudaDeviceSynchronize();
		checkCusolverStatus(status);

		// Allocate worker
		float* workspace;
		cudaMalloc(&workspace, bufferSize * sizeof(float));
		checkCusolverStatus(status);

		int *pivots = nullptr;
		cudaMalloc(&pivots, nRows_ * sizeof(int));

		int *info = nullptr;
		cudaMalloc(&info, sizeof(int));
		cudaMemset(info, 0, sizeof(int));

		// Calculate LU factorization
		status = cusolverDnSgetrf(manager.GetCusolverHandle(),
			nRows_,
			nRows_,
			(float*)LU->buffer->GetPtr(),
			nRows_,
			workspace,
			pivots,
			info);
		cudaDeviceSynchronize();
		checkCusolverStatus(status);

		status = cusolverDnSgetrs(manager.GetCusolverHandle(),
			CUBLAS_OP_N,
			nRows_,
			nRows_,
			(float*)LU->buffer->GetPtr(),
			nRows_,
			pivots,
			(float*)inverse->buffer->GetPtr(),
			nRows_,
			info);
		cudaDeviceSynchronize();
		checkCusolverStatus(status);

		return inverse;
	}

	template<class T>
	__matrix__<T> Matrix<T>::CuTranspose()
	{
		std::shared_ptr<Matrix<T>> C = Matrix<T>::Build(this->nColumns_, this->nRows_, Architecture::Device);

		CudaManager& manager = CudaManager::GetInstance();
		cublasStatus_t status = cublasStatus_t::CUBLAS_STATUS_SUCCESS;
		int leadingDimensionA = this->nRows_;
		int leadingDimensionC = C->nRows();
		int nColC = C->nColumns();
		float alpha = 1.0f;
		float beta = 0.0f;

		status = cublasSgeam(manager.GetCublasHandle(),
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			nRows_,
			nColumns_,
			&alpha,
			(float*)this->buffer->GetPtr(), leadingDimensionA,
			&beta,
			(float*)this->buffer->GetPtr(), leadingDimensionA,
			(float*)C->buffer->GetPtr(), leadingDimensionC);

		checkCudaStatus(status);
		return C;
	}

	// for testing
	template class Matrix<int>;
	template class Matrix<float>;
}

#endif