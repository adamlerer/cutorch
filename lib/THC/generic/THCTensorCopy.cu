#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.cu"
#else

THC_API void
THCTensor_(copy)(THCState* state, THCTensor* dst, THCTensor* src) {
  long totalElements = THCTensor_(nElement)(state, dst);

  THArgCheck(totalElements == THCTensor_(nElement)(state, src), 2,
             "sizes do not match");

  if (THCTensor_(nDimension)(state, dst) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }

  // We can memcpy the memory if:
  // -both tensors are contiguous; or,
  // -there is only one element to copy; or,
  // -FIXME: if both tensors have matching size and stride arrays, and no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is contiguous).
  bool srcContig = THCTensor_(isContiguous)(state, src);
  bool dstContig = THCTensor_(isContiguous)(state, dst);
  bool memcpyEligible = (srcContig && dstContig) || (totalElements == 1);

  int oldDev = curGPU();
  int srcDev = THCTensor_(getDevice)(state, src);
  int dstDev = THCTensor_(getDevice)(state, dst);

  // empirically, running the kernel on the device that holds the
  // non-contiguous tensor is faster by 5-10x
  int copyDev   = dstContig ? srcDev : dstDev;
  int remoteDev = dstContig ? dstDev : srcDev;

  if (srcDev == dstDev) {
    if (oldDev != srcDev) {
      THCudaCheck(cudaSetDevice(srcDev));
    }
  } else {
    // synchronize remote device before copy
    cudaEvent_t dataReady;
    THCudaCheck(cudaSetDevice(remoteDev));
    THCudaCheck(cudaEventCreate(&dataReady));
    THCudaCheck(cudaEventRecord(
                  dataReady,
                  THCState_getDeviceStream(state, remoteDev, THCState_getCurrentStreamIndex(state))));
    THCudaCheck(cudaSetDevice(copyDev));
    THCudaCheck(cudaStreamWaitEvent(
                  THCState_getDeviceStream(state, copyDev, THCState_getCurrentStreamIndex(state)),
                  dataReady, 0));
    THCudaCheck(cudaEventDestroy(dataReady));
  }

  if (memcpyEligible) {
    THCudaCheck(cudaMemcpyAsync(THCTensor_(data)(state, dst),
                                THCTensor_(data)(state, src),
                                totalElements * sizeof(real),
                                cudaMemcpyDeviceToDevice,
                                THCState_getCurrentStream(state)));
  } else {
#ifdef THC_REAL_IS_FLOAT
      bool succ =
        THCTensor_(pointwiseApply2)(state, dst, src, CopyOp<real>());
      THArgCheck(succ, 2, CUTORCH_DIM_WARNING);
#else
#define STRINGIFY(x) #x
      THError("Non-contiguous copy not implemented for Cuda%sTensor", STRINGIFY(Real));
#undef STRINGIFY
#endif
  }

  if (srcDev != dstDev) {
    // synchronize remote device after copy
    cudaEvent_t doneCopying;
    THCudaCheck(cudaEventCreate(&doneCopying));
    THCudaCheck(cudaEventRecord(
                  doneCopying,
                  THCState_getDeviceStream(state, copyDev, THCState_getCurrentStreamIndex(state))));
    THCudaCheck(cudaSetDevice(remoteDev));
    THCudaCheck(cudaStreamWaitEvent(
                  THCState_getDeviceStream(state, remoteDev, THCState_getCurrentStreamIndex(state)),
                  doneCopying, 0));
    THCudaCheck(cudaEventDestroy(doneCopying));
  }

  if (curGPU() != oldDev) {
    THCudaCheck(cudaSetDevice(oldDev));
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

#endif
