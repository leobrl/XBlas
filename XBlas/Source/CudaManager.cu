#include "../Include/CudaManager.cuh"
#include<system_error>
#include<string>

namespace XBlas
{
	CudaManager::CudaManager()
	{
		cublasStatus_t cublasStatus = cublasCreate(&handle);
		if (cublasStatus != CUBLAS_STATUS_SUCCESS)
		{
			std::string message = "CUBLAS initialization failed with error code " + cublasStatus;
			throw std::runtime_error(message);
		}

		cusolverStatus_t cusolverStatus = cusolverDnCreate(&solver_handle);
		if (cusolverStatus != CUSOLVER_STATUS_SUCCESS)
		{
			std::string message = "CUBLAS initialization failed with error code " + cusolverStatus;
			throw std::runtime_error(message);
		}
	}

	CudaManager::~CudaManager()
	{
		cublasDestroy(handle);
	}
}