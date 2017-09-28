#include "../Include/CudaManager.cuh"
#include<system_error>
#include<string>

namespace XBlas
{
	CudaManager::CudaManager()
	{
		cublasStatus_t status = cublasCreate(&handle);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			std::string message = "CUBLAS initialization failed with error code " + status;
			throw std::runtime_error(message);
		}
	}

	CudaManager::~CudaManager()
	{
		cublasDestroy(handle);
	}
}