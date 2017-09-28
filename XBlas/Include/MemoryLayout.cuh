#include <exception>
#include <memory>
#include <cuda_runtime.h>
#include "MemoryBuffer.cuh"
#include "Architecture.hpp"

#ifndef XBLAS_MEMORYLAYOUT
#define XBLAS_MEMORYLAYOUT

namespace XBlas
{
	template<class T>
	class MemoryLayout
	{
		#define BUFFER_PTR (buffer->GetPtr())
	public:
		const virtual size_t Length() const = 0;

	protected:
		std::shared_ptr<MemoryBuffer<T>> buffer;

	public:
		virtual void Move(Architecture arch);

		void inline SetBuffer(const std::shared_ptr<MemoryBuffer<T>>& buffer)
		{
			this->buffer = buffer;
		}
	};

	template<class T>
	void MemoryLayout<T>::Move(Architecture arch)
	{
		this->buffer->Move(arch);
	}
}

#endif