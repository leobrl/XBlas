#include <memory>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"

#ifndef XBLAS_CUDAMANAGER
#define XBLAS_CUDAMANAGER

#define checkCudaStatus(x)	if (x)\
							{\
							std::string message = "Returned status " + x;\
							throw new std::exception(message.c_str());\
							}\

#define checkCusolverStatus(x)	if (x)\
								{\
								std::string message = "Returned status " + x;\
								throw new std::exception(message.c_str());\
								}\

namespace XBlas
{
	class __declspec(dllexport) CudaManager
	{
	private:
		cublasHandle_t handle;
		cusolverDnHandle_t solver_handle;

	public:

		CudaManager(CudaManager const&) = delete;
		CudaManager(CudaManager const&&) = delete;
		CudaManager& operator=(CudaManager const&) = delete;
		CudaManager& operator=(CudaManager const&&) = delete;

		static CudaManager& GetInstance()
		{
			static CudaManager instance;
			return instance;
		};

		const cublasHandle_t GetCublasHandle() const
		{
			return handle;
		}

		const cusolverDnHandle_t GetCusolverHandle() const
		{
			return solver_handle;
		}

		~CudaManager();

	protected:
		CudaManager();
	};
}
#endif
