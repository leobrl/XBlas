#include <memory>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef XBLAS_CUDAMANAGER
#define XBLAS_CUDAMANAGER
namespace XBlas
{
	class __declspec(dllexport) CudaManager
	{
	private:
		cublasHandle_t handle;

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

		~CudaManager();

	protected:
		CudaManager();
	};
}
#endif
