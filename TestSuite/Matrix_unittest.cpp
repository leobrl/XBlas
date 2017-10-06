#include <gtest/gtest.h>
#include<memory>
#include<exception>
#include "Common.h"
#include "../XBlas/Include/Matrix.cuh"

TEST(MatrixTest, Allocation)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::Matrix<float>> matrix = XBlas::Matrix<float>::Build(nRows, nColumns, arch);
}

TEST(MatrixTest, Dimensions)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::Matrix<float>> matrix = XBlas::Matrix<float>::Build(nRows, nColumns, arch);

	ASSERT_EQ(matrix->nRows(), defaultMatrixRows);
	ASSERT_EQ(matrix->nRows(), defaultMatrixColumns);

	for (int col = 0; col < defaultMatrixRows; ++col)
	{
		int length = (matrix->operator[](col))->Length();
		ASSERT_EQ(length, defaultMatrixColumns);
	}
}

TEST(HostMatrixTest, SetGetOperations)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::Matrix<int>> matrix = XBlas::Matrix<int>::Build(nRows, nColumns, arch);

	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			(matrix->operator[](col))->operator[](row) = row + col;
		}
	}

	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			int value = (matrix->operator[](col))->operator[](row);
			EXPECT_EQ(value, row + col);
		}
	}
}

TEST(HostMatrixTest, MoveOperation)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::MemoryBuffer<int>> buffer = XBlas::MemoryBuffer<int>::MemAlloc(nRows*nColumns, arch);
	std::shared_ptr<XBlas::Matrix<int>> matrix = XBlas::Matrix<int>::Build(buffer, nRows, nColumns);

	// Initialize
	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			(matrix->operator[](col))->operator[](row) = row + col;
		}
	}

	// Move to device. 
	int* p = buffer->GetPtr();

	matrix->Move(XBlas::Architecture::Device);

	// Reset memory host side
	for (int element = 0; element < nRows*nColumns; ++element)
	{
		p[element] = 0;
	}

	// Copy back to host from device 
	matrix->Move(XBlas::Architecture::Host);

	// Test
	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			int expected = row + col;
			int value = (matrix->operator[](col))->operator[](row);
			EXPECT_EQ(value, expected);
		}
	}
}

TEST(MatrixTest, MultiplyByScalar)
{
	FAIL();
}

TEST(MatrixTest, MultiplyByVector)
{
	FAIL();
}

TEST(MatrixTest, MultiplyByMatrix)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::Matrix<float>> matrix = XBlas::Matrix<float>::Build(nRows, nColumns, arch);

	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			(matrix->operator[](row))->operator[](col) = 1.0;
		}
	}

	std::shared_ptr<XBlas::Matrix<float>> squaredMatrix = matrix->operator*(matrix);

	double expected = 3.0;
	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			double actual = (squaredMatrix->operator[](col))->operator[](row);
			ASSERT_DOUBLE_EQ(actual, expected);
		}
	}
}

TEST(MatrixTest, Transpose)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::Matrix<int>> matrix = XBlas::Matrix<int>::Build(nRows, nColumns, arch);

	int c = 0;
	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			(matrix->operator[](row))->operator[](col) = c++;
		}
	}

	std::shared_ptr<XBlas::Matrix<int>> transposedMatrix = matrix->Transpose();

	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			int expected = (matrix->operator[](row))->operator[](col);
			int actual = (transposedMatrix->operator[](col))->operator[](row);
			ASSERT_DOUBLE_EQ(actual, expected);
		}
	}

}

TEST(MatrixTest, Identity)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::Matrix<int>> matrix = XBlas::Matrix<int>::Identity(nRows, arch);

	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			int actual = (matrix->operator[](row))->operator[](col);
			if (col == row)
				ASSERT_DOUBLE_EQ(actual, 1);
			else
				ASSERT_DOUBLE_EQ(actual, 0);
		}
	}
}

TEST(MatrixTest, Inverse)
{
	hostDefaultMatrixInput;
	std::shared_ptr<XBlas::Matrix<float>> matrix = XBlas::Matrix<float>::Build(nRows, nColumns, arch);
	int c = 0;
	float values[] = { 0.0, 1.0, -3.0, -3.0, -4.0, 4.0, -2.0, -2.0, 1.0 };
	for (int col = 0; col < nColumns; ++col)
	{
		for (int row = 0; row < nRows; ++row)
		{
			(matrix->operator[](col))->operator[](row) = values[row + col*nColumns];
		}
	}

	std::shared_ptr<XBlas::Matrix<float>> inverse = matrix->Inverse();

	int expected[] = { 4, 5, -8, -5, -6, 9, -2, -2, 3 };
	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nColumns; ++col)
		{
			float actual = (inverse->operator[](col))->operator[](row);
			std::cout << actual << " ";
			//			ASSERT_DOUBLE_EQ(actual, expected[col + row*nRows]);
		}
		std::cout << std::endl;
	}
}