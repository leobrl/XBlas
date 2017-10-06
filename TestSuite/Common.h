#pragma once

#define defaultVectorCapacity 10
#define defaultMatrixRows 3
#define defaultMatrixColumns 3
#define XBlasHost XBlas::Architecture::Host
#define XBlasDevice XBlas::Architecture::Device

#define hostDefaultVectorInput \
	const size_t capacity = defaultVectorCapacity; \
	XBlas::Architecture arch = XBlasHost;

#define deviceDefaultVectorInput \
	const size_t capacity = defaultVectorCapacity; \
	XBlas::Architecture arch = XBlasDevice;

#define hostDefaultMatrixInput \
	const size_t nRows = defaultMatrixRows; \
	const size_t nColumns = defaultMatrixColumns; \
	XBlas::Architecture arch = XBlasHost; //TODO: should be a method

#define deviceDefaultMatrixInput \
	const size_t nRows = defaultMatrixRows; \
	const size_t nColumns = defaultMatrixColumns; \
	XBlas::Architecture arch = XBlasDevice; 
