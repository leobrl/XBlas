#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <exception>
#include "Architecture.hpp"
#include "MemoryBuffer.cuh"
#include "MemoryLayout.cuh"

#ifndef XBLAS_ARRAY
#define XBLAS_ARRAY

namespace XBlas
{
	template<class T>
	class  __declspec(dllexport) Vector : public MemoryLayout<T>
	{
	private:
		
		Vector(size_t capacity, Architecture arch);
		Vector(std::shared_ptr<MemoryBuffer<T>> memoryBuffer);

	public:

		static std::shared_ptr<Vector<T>> Build(long capacity, Architecture arch);
		static std::shared_ptr<Vector<T>> Build(std::shared_ptr<MemoryBuffer<T>> memoryBuffer);

		Vector() = delete;
		Vector(Vector&) = delete;
		Vector(Vector&&) = delete;
		Vector& operator=(Vector const&) = delete;
		Vector& operator=(Vector const&&) = delete;

		T& operator [] (size_t index);
		const T& operator [] (size_t index) const;

		const virtual size_t Length() const
		{
			return buffer->GetCapacity();
		}
	};

	template<class T>
	std::shared_ptr<Vector<T>> Vector<T>::Build(long capacity, Architecture arch)
	{
		if (capacity < 0)
			throw std::out_of_range("Capacity must be a positive number");
		 
		return std::shared_ptr<Vector<T>>(new Vector<T>(capacity, arch));
	}

	template<class T>
	Vector<T>::Vector(size_t capacity, Architecture arch)
	{
		SetBuffer(MemoryBuffer<T>::MemAlloc(capacity, arch));
	}

	template<class T>
	std::shared_ptr<Vector<T>> Vector<T>::Build(std::shared_ptr<MemoryBuffer<T>> memoryBuffer)
	{
		return std::shared_ptr<Vector<T>>(new Vector<T>(memoryBuffer));
	}

	template<class T>
	Vector<T>::Vector(std::shared_ptr<MemoryBuffer<T>> memoryBuffer)
	{
		SetBuffer(memoryBuffer);
	}

	template<class T>
	T & Vector<T>::operator[](size_t index)
	{
		switch (buffer->GetArchitecture())
		{
		case Host:
			return BUFFER_PTR[index];
		case Device:
			INVALID_DEREFERENCING
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}
	}

	template<class T> 
	const T & Vector<T>::operator[](size_t index) const
	{
		switch (buffer->GetArchitecture())
		{
		case Host:
			return BUFFER_PTR[index];
		case Device:
			throw std::domain_error("Cannot dereference device pointer.");
		default:
			throw std::invalid_argument("Allocation attempted with invalid architecture type.");
		}
	}

	// for testing
	template class Vector<int>;
}

#endif // !XBLAS_ARRAY
