#include <gtest/gtest.h>
#include<memory>
#include<exception>
#include "Common.h"
#include "../XBlas/Include/Array.cuh"

TEST(HostArrayTest, SetGetOperations)
{
	hostDefaultArrayInput;
	std::shared_ptr<XBlas::Array<int>> theArray = XBlas::Array<int>::Build(capacity, arch);

	int value = 0;
	while (value < capacity)
	{
		theArray->operator[](value) = value;
		value++;
	};

	value = 0;
	while (value < capacity)
	{
		EXPECT_EQ(theArray->operator[](value), value);
		value++;
	};

}

TEST(DeviceArrayTest, SetGetOperations)
{
	deviceDefaultArrayInput;
	std::shared_ptr<XBlas::Array<int>> theArray = XBlas::Array<int>::Build(capacity, arch);

	EXPECT_THROW(
		theArray->operator[](0) = 1,
		std::domain_error);
}