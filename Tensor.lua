function torch.CudaTensor.apply(self, func)
   local x = torch.FloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
   return self
end

local function Tensor__type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end
local function Tensor__typeAs(self,tensor)
   return self:type(tensor:type())
end

local TensorTypes = {
   float  = 'torch.FloatTensor',
   double = 'torch.DoubleTensor',
   byte   = 'torch.ByteTensor',
   char   = 'torch.CharTensor',
   int    = 'torch.IntTensor',
   short  = 'torch.ShortTensor',
   long   = 'torch.LongTensor'
}

local CudaTensorTypes = {
   cuda       = 'torch.CudaTensor',
   cudaDouble = 'torch.CudaDoubleTensor',
   cudaByte   = 'torch.CudaByteTensor',
   cudaChar   = 'torch.CudaCharTensor',
   cudaInt    = 'torch.CudaIntTensor',
   cudaShort  = 'torch.CudaShortTensor',
   cudaLong   = 'torch.CudaLongTensor'
}

function Tensor__converter(type)
    return function(self)
        return self:type(type)
    end
end

-- CPU -> CUDA
for _, TensorType in pairs(TensorTypes) do
    for DataType, CudaTensorType in pairs(CudaTensorTypes) do
        rawset(torch.getmetatable(TensorType), DataType, Tensor__converter(CudaTensorType))
    end
end

-- CUDA -> CPU
for _, CudaTensorType in pairs(CudaTensorTypes) do
    rawset(torch.getmetatable(CudaTensorType), 'type', Tensor__type)
    rawset(torch.getmetatable(CudaTensorType), 'typeAs', Tensor__typeAs)
    for DataType, TensorType in pairs(TensorTypes) do
       rawset(torch.getmetatable(CudaTensorType), DataType, Tensor__converter(TensorType))
    end
end

-- FIXME: CUDA -> CUDA not yet implemented
-- odd man out
rawset(torch.getmetatable('torch.CudaTensor'), 'cuda', Tensor__converter('torch.CudaTensor'))

--FIXME add conversions from CUDA types to other CUDA types
--
do
    local metatable = torch.getmetatable('torch.CudaTensor')
    for _,func in pairs{'expand', 'expandAs', 'view', 'viewAs', 'repeatTensor',
                        'permute', 'split', 'chunk'} do
        rawset(metatable, func, torch[func])
    end
end
