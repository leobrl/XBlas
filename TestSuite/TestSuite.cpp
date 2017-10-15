
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <gtest/gtest.h>
#include <stdio.h>

#include<iostream>
#include <gtest/gtest.h>

int main(int ac, char* av[])
{
	testing::InitGoogleTest(&ac, av);
	testing::GTEST_FLAG(filter) = "*";
	RUN_ALL_TESTS();
}

