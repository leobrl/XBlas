#pragma once

#define defaultArrayCapacity 10
#define defaultMatrixRows 3
#define defaultMatrixColumns 3
#define XBlasHost XBlas::Architecture::Host
#define XBlasDevice XBlas::Architecture::Device

#define hostDefaultArrayInput \
	const size_t capacity = defaultArrayCapacity; \
	XBlas::Architecture arch = XBlasHost;

#define deviceDefaultArrayInput \
	const size_t capacity = defaultArrayCapacity; \
	XBlas::Architecture arch = XBlasDevice;

#define hostDefaultMatrixInput \
	const size_t nRows = defaultMatrixRows; \
	const size_t nColumns = defaultMatrixColumns; \
	XBlas::Architecture arch = XBlasHost; //TODO: should be a method

#define deviceDefaultMatrixInput \
	const size_t nRows = defaultMatrixRows; \
	const size_t nColumns = defaultMatrixColumns; \
	XBlas::Architecture arch = XBlasDevice; 
