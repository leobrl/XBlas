#include <gtest/gtest.h>
#include<memory>
#include<exception>
#include "Common.h"
#include "../XBlas/Include/Vector.cuh"

TEST(HostVectorTest, SetGetOperations)
{
	hostDefaultVectorInput;
	std::shared_ptr<XBlas::Vector<int>> theVector = XBlas::Vector<int>::Build(capacity, arch);

	int value = 0;
	while (value < capacity)
	{
		theVector->operator[](value) = value;
		value++;
	};

	value = 0;
	while (value < capacity)
	{
		EXPECT_EQ(theVector->operator[](value), value);
		value++;
	};

}

TEST(DeviceVectorTest, SetGetOperations)
{
	deviceDefaultVectorInput;
	std::shared_ptr<XBlas::Vector<int>> theVector = XBlas::Vector<int>::Build(capacity, arch);

	EXPECT_THROW(
		theVector->operator[](0) = 1,
		std::domain_error);
}