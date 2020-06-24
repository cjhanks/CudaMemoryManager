#include "error.hh"

namespace cmm {
Error::Error(std::string message)
  : std::runtime_error(message) {}

void
Error::Throw(cudaError_t rc)
{
  std::string msg;
  switch (rc) {
    case cudaSuccess:
      msg = "cudaSuccess";
      break;
    case cudaErrorInvalidValue:
      msg = "cudaErrorInvalidValue";
      break;
    case cudaErrorMemoryAllocation:
      msg = "cudaErrorMemoryAllocation";
      break;
    case cudaErrorInitializationError:
      msg = "cudaErrorInitializationError";
      break;
    case cudaErrorCudartUnloading:
      msg = "cudaErrorCudartUnloading";
      break;
    case cudaErrorProfilerDisabled:
      msg = "cudaErrorProfilerDisabled";
      break;
    case cudaErrorProfilerNotInitialized:
      msg = "cudaErrorProfilerNotInitialized";
      break;
    case cudaErrorProfilerAlreadyStarted:
      msg = "cudaErrorProfilerAlreadyStarted";
      break;
    case cudaErrorProfilerAlreadyStopped:
      msg = "cudaErrorProfilerAlreadyStopped";
      break;
    case cudaErrorInvalidConfiguration:
      msg = "cudaErrorInvalidConfiguration";
      break;
    case cudaErrorInvalidPitchValue:
      msg = "cudaErrorInvalidPitchValue";
      break;
    case cudaErrorInvalidSymbol:
      msg = "cudaErrorInvalidSymbol";
      break;
    case cudaErrorInvalidHostPointer:
      msg = "cudaErrorInvalidHostPointer";
      break;
    case cudaErrorInvalidDevicePointer:
      msg = "cudaErrorInvalidDevicePointer";
      break;
    case cudaErrorInvalidTexture:
      msg = "cudaErrorInvalidTexture";
      break;
    case cudaErrorInvalidTextureBinding:
      msg = "cudaErrorInvalidTextureBinding";
      break;
    case cudaErrorInvalidChannelDescriptor:
      msg = "cudaErrorInvalidChannelDescriptor";
      break;
    case cudaErrorInvalidMemcpyDirection:
      msg = "cudaErrorInvalidMemcpyDirection";
      break;
    case cudaErrorAddressOfConstant:
      msg = "cudaErrorAddressOfConstant";
      break;
    case cudaErrorTextureFetchFailed:
      msg = "cudaErrorTextureFetchFailed";
      break;
    case cudaErrorTextureNotBound:
      msg = "cudaErrorTextureNotBound";
      break;
    case cudaErrorSynchronizationError:
      msg = "cudaErrorSynchronizationError";
      break;
    case cudaErrorInvalidFilterSetting:
      msg = "cudaErrorInvalidFilterSetting";
      break;
    case cudaErrorInvalidNormSetting:
      msg = "cudaErrorInvalidNormSetting";
      break;
    case cudaErrorMixedDeviceExecution:
      msg = "cudaErrorMixedDeviceExecution";
      break;
    case cudaErrorNotYetImplemented:
      msg = "cudaErrorNotYetImplemented";
      break;
    case cudaErrorMemoryValueTooLarge:
      msg = "cudaErrorMemoryValueTooLarge";
      break;
    case cudaErrorInsufficientDriver:
      msg = "cudaErrorInsufficientDriver";
      break;
    case cudaErrorInvalidSurface:
      msg = "cudaErrorInvalidSurface";
      break;
    case cudaErrorDuplicateVariableName:
      msg = "cudaErrorDuplicateVariableName";
      break;
    case cudaErrorDuplicateTextureName:
      msg = "cudaErrorDuplicateTextureName";
      break;
    case cudaErrorDuplicateSurfaceName:
      msg = "cudaErrorDuplicateSurfaceName";
      break;
    case cudaErrorDevicesUnavailable:
      msg = "cudaErrorDevicesUnavailable";
      break;
    case cudaErrorIncompatibleDriverContext:
      msg = "cudaErrorIncompatibleDriverContext";
      break;
    case cudaErrorMissingConfiguration:
      msg = "cudaErrorMissingConfiguration";
      break;
    case cudaErrorPriorLaunchFailure:
      msg = "cudaErrorPriorLaunchFailure";
      break;
    case cudaErrorLaunchMaxDepthExceeded:
      msg = "cudaErrorLaunchMaxDepthExceeded";
      break;
    case cudaErrorLaunchFileScopedTex:
      msg = "cudaErrorLaunchFileScopedTex";
      break;
    case cudaErrorLaunchFileScopedSurf:
      msg = "cudaErrorLaunchFileScopedSurf";
      break;
    case cudaErrorSyncDepthExceeded:
      msg = "cudaErrorSyncDepthExceeded";
      break;
    case cudaErrorLaunchPendingCountExceeded:
      msg = "cudaErrorLaunchPendingCountExceeded";
      break;
    case cudaErrorInvalidDeviceFunction:
      msg = "cudaErrorInvalidDeviceFunction";
      break;
    case cudaErrorNoDevice:
      msg = "cudaErrorNoDevice";
      break;
    case cudaErrorInvalidDevice:
      msg = "cudaErrorInvalidDevice";
      break;
    case cudaErrorStartupFailure:
      msg = "cudaErrorStartupFailure";
      break;
    case cudaErrorInvalidKernelImage:
      msg = "cudaErrorInvalidKernelImage";
      break;
    //case cudaErrorDeviceUninitialized:
    //  msg = "cudaErrorDeviceUninitialized";
    //  break;
    case cudaErrorMapBufferObjectFailed:
      msg = "cudaErrorMapBufferObjectFailed";
      break;
    case cudaErrorUnmapBufferObjectFailed:
      msg = "cudaErrorUnmapBufferObjectFailed";
      break;
    case cudaErrorArrayIsMapped:
      msg = "cudaErrorArrayIsMapped";
      break;
    case cudaErrorAlreadyMapped:
      msg = "cudaErrorAlreadyMapped";
      break;
    case cudaErrorNoKernelImageForDevice:
      msg = "cudaErrorNoKernelImageForDevice";
      break;
    case cudaErrorAlreadyAcquired:
      msg = "cudaErrorAlreadyAcquired";
      break;
    case cudaErrorNotMapped:
      msg = "cudaErrorNotMapped";
      break;
    case cudaErrorNotMappedAsArray:
      msg = "cudaErrorNotMappedAsArray";
      break;
    case cudaErrorNotMappedAsPointer:
      msg = "cudaErrorNotMappedAsPointer";
      break;
    case cudaErrorECCUncorrectable:
      msg = "cudaErrorECCUncorrectable";
      break;
    case cudaErrorUnsupportedLimit:
      msg = "cudaErrorUnsupportedLimit";
      break;
    case cudaErrorDeviceAlreadyInUse:
      msg = "cudaErrorDeviceAlreadyInUse";
      break;
    case cudaErrorPeerAccessUnsupported:
      msg = "cudaErrorPeerAccessUnsupported";
      break;
    case cudaErrorInvalidPtx:
      msg = "cudaErrorInvalidPtx";
      break;
    case cudaErrorInvalidGraphicsContext:
      msg = "cudaErrorInvalidGraphicsContext";
      break;
    case cudaErrorNvlinkUncorrectable:
      msg = "cudaErrorNvlinkUncorrectable";
      break;
    case cudaErrorJitCompilerNotFound:
      msg = "cudaErrorJitCompilerNotFound";
      break;
    case cudaErrorInvalidSource:
      msg = "cudaErrorInvalidSource";
      break;
    case cudaErrorFileNotFound:
      msg = "cudaErrorFileNotFound";
      break;
    case cudaErrorSharedObjectSymbolNotFound:
      msg = "cudaErrorSharedObjectSymbolNotFound";
      break;
    case cudaErrorSharedObjectInitFailed:
      msg = "cudaErrorSharedObjectInitFailed";
      break;
    case cudaErrorOperatingSystem:
      msg = "cudaErrorOperatingSystem";
      break;
    case cudaErrorInvalidResourceHandle:
      msg = "cudaErrorInvalidResourceHandle";
      break;
    case cudaErrorIllegalState:
      msg = "cudaErrorIllegalState";
      break;
    case cudaErrorSymbolNotFound:
      msg = "cudaErrorSymbolNotFound";
      break;
    case cudaErrorNotReady:
      msg = "cudaErrorNotReady";
      break;
    case cudaErrorIllegalAddress:
      msg = "cudaErrorIllegalAddress";
      break;
    case cudaErrorLaunchOutOfResources:
      msg = "cudaErrorLaunchOutOfResources";
      break;
    case cudaErrorLaunchIncompatibleTexturing:
      msg = "cudaErrorLaunchIncompatibleTexturing";
      break;
    case cudaErrorPeerAccessAlreadyEnabled:
      msg = "cudaErrorPeerAccessAlreadyEnabled";
      break;
    case cudaErrorPeerAccessNotEnabled:
      msg = "cudaErrorPeerAccessNotEnabled";
      break;
    case cudaErrorSetOnActiveProcess:
      msg = "cudaErrorSetOnActiveProcess";
      break;
    case cudaErrorContextIsDestroyed:
      msg = "cudaErrorContextIsDestroyed";
      break;
    case cudaErrorAssert:
      msg = "cudaErrorAssert";
      break;
    case cudaErrorTooManyPeers:
      msg = "cudaErrorTooManyPeers";
      break;
    case cudaErrorHostMemoryAlreadyRegistered:
      msg = "cudaErrorHostMemoryAlreadyRegistered";
      break;
    case cudaErrorHostMemoryNotRegistered:
      msg = "cudaErrorHostMemoryNotRegistered";
      break;
    case cudaErrorHardwareStackError:
      msg = "cudaErrorHardwareStackError";
      break;
    case cudaErrorIllegalInstruction:
      msg = "cudaErrorIllegalInstruction";
      break;
    case cudaErrorMisalignedAddress:
      msg = "cudaErrorMisalignedAddress";
      break;
    case cudaErrorInvalidAddressSpace:
      msg = "cudaErrorInvalidAddressSpace";
      break;
    case cudaErrorInvalidPc:
      msg = "cudaErrorInvalidPc";
      break;
    case cudaErrorLaunchFailure:
      msg = "cudaErrorLaunchFailure";
      break;
    case cudaErrorCooperativeLaunchTooLarge:
      msg = "cudaErrorCooperativeLaunchTooLarge";
      break;
    case cudaErrorNotSupported:
      msg = "cudaErrorNotSupported";
      break;
    case cudaErrorSystemNotReady:
      msg = "cudaErrorSystemNotReady";
      break;
    case cudaErrorSystemDriverMismatch:
      msg = "cudaErrorSystemDriverMismatch";
      break;
    case cudaErrorCompatNotSupportedOnDevice:
      msg = "cudaErrorCompatNotSupportedOnDevice";
      break;
    case cudaErrorStreamCaptureUnsupported:
      msg = "cudaErrorStreamCaptureUnsupported";
      break;
    case cudaErrorStreamCaptureInvalidated:
      msg = "cudaErrorStreamCaptureInvalidated";
      break;
    case cudaErrorStreamCaptureMerge:
      msg = "cudaErrorStreamCaptureMerge";
      break;
    case cudaErrorStreamCaptureUnmatched:
      msg = "cudaErrorStreamCaptureUnmatched";
      break;
    case cudaErrorStreamCaptureUnjoined:
      msg = "cudaErrorStreamCaptureUnjoined";
      break;
    case cudaErrorStreamCaptureIsolation:
      msg = "cudaErrorStreamCaptureIsolation";
      break;
    case cudaErrorStreamCaptureImplicit:
      msg = "cudaErrorStreamCaptureImplicit";
      break;
    case cudaErrorCapturedEvent:
      msg = "cudaErrorCapturedEvent";
      break;
    case cudaErrorStreamCaptureWrongThread:
      msg = "cudaErrorStreamCaptureWrongThread";
      break;
    //case cudaErrorTimeout:
    //  msg = "cudaErrorTimeout";
    //  break;
    //case cudaErrorGraphExecUpdateFailure:
    //  msg = "cudaErrorGraphExecUpdateFailure";
    //  break;
    case cudaErrorUnknown:
      msg = "cudaErrorUnknown";
      break;
    case cudaErrorApiFailureBase:
      msg = "cudaErrorApiFailureBase";
      break;
    default:
      msg = "cudaErrorUnknown";
      break;
  }

  msg = msg + " [" + std::to_string(rc) + "]";
  throw Error(msg);
}

// -------------------------------------------------------------------------- //

Canary::~Canary()
{
  Error::Check(cudaGetLastError());
}

} // ns cmm
