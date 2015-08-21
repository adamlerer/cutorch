#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.cu"
#else

void THCStorage_(rawCopy)(THCState *state, THCStorage *self, real *src)
{
  THCudaCheck(cudaMemcpyAsync(self->data, src, self->size * sizeof(real), cudaMemcpyDeviceToDevice, THCState_getCurrentStream(state)));
}

void THCStorage_(copy)(THCState *state, THCStorage *self, THCStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(real), cudaMemcpyDeviceToDevice, THCState_getCurrentStream(state)));
}

void THCStorage_(copyCuda)(THCState *state, THCStorage *self, THCStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(real), cudaMemcpyDeviceToDevice, THCState_getCurrentStream(state)));
}

#endif
