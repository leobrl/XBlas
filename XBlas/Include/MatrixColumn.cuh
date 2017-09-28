#include <cuda_runtime.h>
#include <memory>
#include "Array.cuh"

#ifndef XBLAS_MATRIX_COLUMN
#define XBLAS_MATRIX_COLUMN
namespace XBlas
{
	template<class T>
	class __declspec(dllexport) MatrixColumn
	{
	private:
		T* p;
		size_t length;
		Architecture arch;

		MatrixColumn(T* p, const size_t length, Architecture arch);

	public:

		static std::shared_ptr<MatrixColumn<T>> Build(T* const p, const size_t length, Architecture arch);

		const virtual size_t Length() const
		{
			return length;
		}

		T & operator[](size_t index)
		{
			switch (arch)
			{
			case Host:
				return p[index];
				break;
			case Device:
				INVALID_DEREFERENCING
			default:
				INVALID_ARCHITECTURE_EXCEPTION
			}
		}

		const T & operator[](size_t index) const
		{
			switch (arch)
			{
			case Host:
				return p[index]
					break;
			case Device:
				INVALID_DEREFERENCING
			default:
				INVALID_ARCHITECTURE_EXCEPTION
			}
		}
	
		void Move(T* const p, Architecture arch);
	};

	template<class T>
	std::shared_ptr<MatrixColumn<T>> MatrixColumn<T>::Build(T* const p, const size_t length, Architecture arch)
	{
		return std::shared_ptr<MatrixColumn<T>>(new MatrixColumn<T>(p, length, arch));
	}

	template<class T>
	MatrixColumn<T>::MatrixColumn(T* p, const size_t length, Architecture arch)
		: p{ p }, length{ length }, arch{arch}
	{
	}

	template<class T>
	void MatrixColumn<T>::Move(T* const p, Architecture arch)
	{
		this->p = p;
		this->arch = arch;
	}
}
#endif