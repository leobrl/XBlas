#include <gtest/gtest.h>
#include<memory>
#include<exception>
#include "Common.h"
#include "../XBlas/Include/MemoryBuffer.cuh"

TEST(MemoryBuffer_hostTest, Allocation)
{
	hostDefaultVectorInput;
	std::shared_ptr<XBlas::MemoryBuffer<int>> theArray = XBlas::MemoryBuffer<int>::MemAlloc(capacity, arch);
}

TEST(MemoryBuffer_hostTest, AllocationNegativeSize)
{
	hostDefaultVectorInput;
	EXPECT_THROW(
		XBlas::MemoryBuffer<int>::MemAlloc(-capacity, arch),
		std::out_of_range
	);
}

TEST(MemoryBuffer_deviceTest, Allocation)
{
	deviceDefaultVectorInput;
	std::shared_ptr<XBlas::MemoryBuffer<int>> theArray = XBlas::MemoryBuffer<int>::MemAlloc(capacity, arch);
}

TEST(MemoryBuffer_deviceTest, AllocationNegativeSize)
{
	deviceDefaultVectorInput;
	EXPECT_THROW(
		XBlas::MemoryBuffer<int>::MemAlloc(-capacity, arch),
		std::out_of_range
	);
}

TEST(MemoryBuffer_deviceTest, MoveOperation)
{
	hostDefaultVectorInput;
	std::shared_ptr<XBlas::MemoryBuffer<int>> buffer = XBlas::MemoryBuffer<int>::MemAlloc(capacity, arch);

	int *p1 = buffer->GetPtr();
	int value = 0;
	while (value < capacity)
	{
		*p1 = value;
		p1++;
		value++;
	};

	int *p2 = buffer->GetPtr();

	buffer->Move(XBlas::Architecture::Device);
	
	while (value < capacity)
	{
		*p2 = 0;
		p2++;
	};
	
	buffer->Move(XBlas::Architecture::Host);

	int* p = buffer->GetPtr();
	value = 0;
	while (value < capacity)
	{
		EXPECT_EQ(*p, value);
		p++;
		value++;
	};
}
