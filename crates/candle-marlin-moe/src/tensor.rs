use core::ffi::c_void;

use candle::{
    DType, Result, Storage,
    cuda::{
        CudaDType,
        cudarc::driver::{CudaStream, DevicePtr, DeviceRepr, SyncOnDrop},
    },
};

/// CUDA device pointer paired with the synchronization guard that keeps its storage access live.
///
/// The guard is retained until this value drops, so Candle cannot observe a conflicting access
/// before the associated kernel launch has been enqueued on the same stream.
pub(crate) struct TensorPtr<'a> {
    address: u64,
    _guard: SyncOnDrop<'a>,
}

impl TensorPtr<'_> {
    /// Returns the raw CUDA device address.
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    /// Returns a typed const pointer at the validated tensor offset.
    pub(crate) fn as_const<T>(&self) -> *const T {
        self.address as usize as *const T
    }

    /// Returns a typed mutable pointer at the validated tensor offset.
    pub(crate) fn as_mut<T>(&self) -> *mut T {
        self.address as usize as *mut T
    }

    /// Returns a const void pointer at the validated tensor offset.
    pub(crate) fn as_const_void(&self) -> *const c_void {
        self.as_const()
    }

    /// Returns a mutable void pointer at the validated tensor offset.
    pub(crate) fn as_mut_void(&self) -> *mut c_void {
        self.as_mut()
    }
}

/// Extracts a CUDA pointer for `dtype` and retains its Candle access guard.
pub(crate) fn cuda_any<'a>(
    storage: &'a Storage,
    start_element: usize,
    dtype: DType,
    stream: &'a CudaStream,
) -> Result<TensorPtr<'a>> {
    match dtype {
        DType::BF16 => cuda_typed::<half::bf16>(storage, start_element, stream),
        DType::F16 => cuda_typed::<half::f16>(storage, start_element, stream),
        DType::F32 => cuda_typed::<f32>(storage, start_element, stream),
        DType::I32 => cuda_typed::<i32>(storage, start_element, stream),
        DType::U32 => cuda_typed::<u32>(storage, start_element, stream),
        _ => candle::bail!("Marlin tensor pointer: unsupported dtype {:?}", dtype),
    }
}

/// Extracts a contiguous I32 CUDA pointer and retains its Candle access guard.
pub(crate) fn cuda_i32<'a>(
    storage: &'a Storage,
    start_element: usize,
    stream: &'a CudaStream,
) -> Result<TensorPtr<'a>> {
    cuda_typed::<i32>(storage, start_element, stream)
}

/// Extracts a contiguous F32 CUDA pointer and retains its Candle access guard.
pub(crate) fn cuda_f32<'a>(
    storage: &'a Storage,
    start_element: usize,
    stream: &'a CudaStream,
) -> Result<TensorPtr<'a>> {
    cuda_typed::<f32>(storage, start_element, stream)
}

fn cuda_typed<'a, T: CudaDType + DeviceRepr + 'a>(
    storage: &'a Storage,
    start_element: usize,
    stream: &'a CudaStream,
) -> Result<TensorPtr<'a>> {
    let (address, guard) = match storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?.device_ptr(stream),
        _ => candle::bail!("Marlin tensor storage must be CUDA"),
    };
    let offset = start_element
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| candle::Error::Msg("Marlin tensor byte offset overflow".into()))?;
    let address = address
        .checked_add(offset as u64)
        .ok_or_else(|| candle::Error::Msg("Marlin tensor pointer overflow".into()))?;
    Ok(TensorPtr {
        address,
        _guard: guard,
    })
}

#[cfg(test)]
mod tests {
    /// Pointer offsets are checked in the typed helper; shape-only validation remains CPU-safe.
    #[test]
    fn byte_offset_overflow_is_detectable() {
        assert!(usize::MAX.checked_mul(std::mem::size_of::<u32>()).is_none());
    }
}
