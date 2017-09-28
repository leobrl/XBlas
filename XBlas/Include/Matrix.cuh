#include <cuda_runtime.h>
#include <exception>
#include <vector>
#include <string>
#include "CudaManager.cuh"
#include "Architecture.hpp"
#include "MemoryLayout.cuh"
#include "MatrixColumn.cuh"


#ifndef XBLAS_MATRIX
#define XBLAS_MATRIX

namespace XBlas
{
	template<class T>
	class  __declspec(dllexport) Matrix : public MemoryLayout<T>
	{
		using __matrix__ = std::shared_ptr<Matrix<T>>;
		using __buffer__ = std::shared_ptr<MemoryBuffer<T>>;
		using __column__ = std::shared_ptr<MatrixColumn<T>>;

	private:

		size_t nRows_;
		size_t nColumns_;
		std::vector<__column__> columns;

		Matrix(size_t nRows, size_t nColumns, Architecture arch);
		Matrix(__buffer__ memoryBuffer, long nRows, long nColumns);

		void SetColumns();

	public:

		Matrix() = delete;
		Matrix(Matrix&) = delete;
		Matrix(Matrix&&) = delete;
		Matrix& operator=(Matrix const&) = delete;
		Matrix& operator=(Matrix const&&) = delete;

		static __matrix__ Build(long nRows, long nColumns, Architecture arch);

		static __matrix__ Build(__buffer__ memoryBuffer, long nRows, long nColumns);

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

		__column__& operator [] (size_t index);

		const __column__& operator [] (size_t index) const;

		std::shared_ptr<Matrix<T>> operator* (std::shared_ptr<Matrix<T>> B);

		void Transpose();

		void Inverse();
	};

	template<class T>
	std::shared_ptr<Matrix<T>> Matrix<T>::Build(long nRows, long nColumns, Architecture arch)
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
	std::shared_ptr<Matrix<T>> Matrix<T>::Build(__buffer__ memoryBuffer, long nRows, long nColumns)
	{
		if (nRows*nColumns < 0)
			throw std::out_of_range("Marix dimensions must be positive numbers");

		return std::shared_ptr<Matrix<T>>(new Matrix<T>(memoryBuffer, nRows, nColumns));
	}

	template<class T>
	Matrix<T>::Matrix(__buffer__ memoryBuffer, long nRows, long nColumns)
	{
		if (nRows*nColumns < 0)
			throw std::out_of_range("Marix dimensions must be positive numbers");

		nRows_ = nRows;
		nColumns_ = nColumns;

		SetBuffer(memoryBuffer);
		SetColumns();
	}

	template<class T>
	void Matrix<T>::Move(Architecture arch)
	{
		MemoryLayout<T>::Move(arch);

		T* colPtr{ buffer->GetPtr() };
		for (size_t col = 0; col < nColumns_; ++col)
		{
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
	std::shared_ptr<Matrix<T>> Matrix<T>::operator* (std::shared_ptr<Matrix<T>> B)
	{
		//TODO refactor this
		//TODO: check return of sgemm

		if (sizeof(T) > 4)
			throw std::invalid_argument("Not implemented");

		std::shared_ptr<Matrix<T>> C = Matrix<T>::Build(this->nRows_, B->nColumns(), Architecture::Device);

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

			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}

		if (status)
		{
			std::string message = "Sgemm returned status " + status;
			throw new std::exception(message.c_str());
		}
		return C;
	}

	template<class T>
	void Matrix<T>::Transpose()
	{

	}

	template<class T>
	void Matrix<T>::Inverse()
	{

	}

	// for testing
	template class Matrix<int>;
}

#endif