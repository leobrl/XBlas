#include <exception>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include "Architecture.hpp"

#ifndef XBLAS_MEMORYBUFFER
#define XBLAS_MEMORYBUFFER

namespace XBlas
{
	#define INVALID_ARCHITECTURE_EXCEPTION \
		throw std::invalid_argument("Allocation attempted with invalid architecture type.");
	#define INVALID_DEREFERENCING \
		throw std::domain_error("Cannot dereference device pointer.");


	template<class T>
	class MemoryBuffer
	{
	private:
		T* h_ptr{ nullptr };
		T* d_ptr{ nullptr };
		Architecture arch;
		size_t capacity;

		// TODO: remove this by using smart pointers for h_ptr and d_ptr. 
		// Requires implementation of smart pointer for CUDA
		bool requirePtrDeletion{ true }; 

		MemoryBuffer(size_t capacity, Architecture arch);
		MemoryBuffer(size_t capacity, Architecture arch, T* ptr);
		void Alloc(size_t capacity, Architecture arch);

	public:
		MemoryBuffer() = delete;
		MemoryBuffer(MemoryBuffer&) = delete;
		MemoryBuffer(MemoryBuffer&&) = delete;
		MemoryBuffer& operator=(MemoryBuffer const&) = delete;
		MemoryBuffer& operator=(MemoryBuffer const&&) = delete;

		~MemoryBuffer();

		inline T* GetPtr();
		inline Architecture GetArchitecture();
		inline size_t GetCapacity();

		void MemoryBuffer<T>::Move(Architecture arch);

		static std::shared_ptr<MemoryBuffer<T>> MemAlloc(long capacity, Architecture arch);
		static std::shared_ptr<MemoryBuffer<T>> MemSet(long capacity, Architecture arch, T* ptr);
	};

	template<class T>
	MemoryBuffer<T>::MemoryBuffer(size_t capacity, Architecture arch)
		: capacity{ capacity }, arch{ arch }
	{
		this->Alloc(capacity, arch);
	}

	template<class T>
	MemoryBuffer<T>::MemoryBuffer(size_t capacity, Architecture arch, T* ptr)
		: capacity{ capacity }, arch{ arch }
	{
		switch (arch)
		{
		case Host:
			h_ptr = ptr;
			break;
		case Device:
			d_ptr = ptr;
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}

		requirePtrDeletion = false;
	}

	template<class T>
	MemoryBuffer<T>::~MemoryBuffer()
	{
		if (requirePtrDeletion)
		{
			if (h_ptr) delete[] h_ptr;
			if (d_ptr) cudaFree(d_ptr);
		}
	}

	template<class T>
	std::shared_ptr<MemoryBuffer<T>> MemoryBuffer<T>::MemAlloc(long capacity, Architecture arch)
	{
		if (capacity < 0)
			throw std::out_of_range("Capacity must be a positive number");

		return std::shared_ptr<MemoryBuffer<T>>(new MemoryBuffer<T>(capacity, arch));
	}

	template<class T>
	std::shared_ptr<MemoryBuffer<T>> MemoryBuffer<T>::MemSet(long capacity, Architecture arch, T* ptr)
	{
		if (capacity < 0)
			throw std::out_of_range("Capacity must be a positive number");

		return std::shared_ptr<MemoryBuffer<T>>(new MemoryBuffer<T>(capacity, arch, ptr));
	}

	template<class T>
	void MemoryBuffer<T>::Alloc(size_t capacity, Architecture arch)
	{
		switch (arch)
		{
		case Host:
			h_ptr = new T[capacity];
			break;
		case Device:
			cudaError_t err;
			err = cudaMalloc((void**)&d_ptr, capacity * sizeof(T));
			if (err != 0)
			{
				std::string message = "cudaMemcpy error code: " + std::to_string(err);
				throw std::exception(message.c_str());
			}
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}
	}

	template<class T>
	void MemoryBuffer<T>::Move(Architecture arch)
	{
		if (this->arch == arch) return;
		this->arch = arch;

		cudaError_t err;
		switch (arch)
		{
		case Host:
			if (!h_ptr) {
				this->Alloc(capacity, arch);
			}
			err = cudaMemcpy(h_ptr, d_ptr, capacity * sizeof(T), cudaMemcpyDeviceToHost);
			break;
		case Device:
			if (!d_ptr) {
				this->Alloc(capacity, arch);
			}
			err = cudaMemcpy(d_ptr, h_ptr, capacity * sizeof(T), cudaMemcpyHostToDevice);
			break;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}

		if (err != 0)
		{
			std::string message = "cudaMemcpy error code: " + std::to_string(err);
			throw std::exception(message.c_str());
		}
	}

	template<class T>
	inline T* MemoryBuffer<T>::GetPtr()
	{
		switch (arch)
		{
		case Host:
			return h_ptr;
		case Device:
			return d_ptr;
		default:
			INVALID_ARCHITECTURE_EXCEPTION
		}
	}

	template<class T>
	inline Architecture MemoryBuffer<T>::GetArchitecture()
	{
		return this->arch;
	}

	template<class T>
	inline size_t MemoryBuffer<T>::GetCapacity()
	{
		return this->capacity;
	}
}

#endif