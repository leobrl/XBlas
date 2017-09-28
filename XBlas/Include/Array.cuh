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
	class  __declspec(dllexport) Array : public MemoryLayout<T>
	{
	private:
		
		Array(size_t capacity, Architecture arch);
		Array(std::shared_ptr<MemoryBuffer<T>> memoryBuffer);

	public:

		static std::shared_ptr<Array<T>> Build(long capacity, Architecture arch);
		static std::shared_ptr<Array<T>> Build(std::shared_ptr<MemoryBuffer<T>> memoryBuffer);

		Array() = delete;
		Array(Array&) = delete;
		Array(Array&&) = delete;
		Array& operator=(Array const&) = delete;
		Array& operator=(Array const&&) = delete;

		T& operator [] (size_t index);
		const T& operator [] (size_t index) const;

		const virtual size_t Length() const
		{
			return buffer->GetCapacity();
		}
	};

	template<class T>
	std::shared_ptr<Array<T>> Array<T>::Build(long capacity, Architecture arch)
	{
		if (capacity < 0)
			throw std::out_of_range("Capacity must be a positive number");
		 
		return std::shared_ptr<Array<T>>(new Array<T>(capacity, arch));
	}

	template<class T>
	Array<T>::Array(size_t capacity, Architecture arch)
	{
		SetBuffer(MemoryBuffer<T>::MemAlloc(capacity, arch));
	}

	template<class T>
	std::shared_ptr<Array<T>> Array<T>::Build(std::shared_ptr<MemoryBuffer<T>> memoryBuffer)
	{
		return std::shared_ptr<Array<T>>(new Array<T>(memoryBuffer));
	}

	template<class T>
	Array<T>::Array(std::shared_ptr<MemoryBuffer<T>> memoryBuffer)
	{
		SetBuffer(memoryBuffer);
	}

	template<class T>
	T & Array<T>::operator[](size_t index)
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
	const T & Array<T>::operator[](size_t index) const
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
	template class Array<int>;
}

#endif // !XBLAS_ARRAY
