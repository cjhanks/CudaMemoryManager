#ifndef CMM_TOOLS_SCOPE_HH_
#define CMM_TOOLS_SCOPE_HH_


namespace cmm {
///
/// @class SyncScope
///
/// On the call to the destructor, the current thread local cuda stream will be
/// synchronized.
///
/// This *can* be used as a flow control tool, but it is primarily meant to be
/// used as tool for scaffolding an application before performance optimization.
///
class SyncScope {
 public:
  ~SyncScope();
};

///
/// @class Canary
///
/// A tiny class you can drop in to a function which checks and throws an
/// exception from the destructor.  Exceptions thrown by this class are *not*
/// meant to be caught.  This is a debugging tool to help you trackdown
/// unchecked cuda return codes.
///
class Canary {
 public:
  Canary();
  ~Canary();

  void
  Check();
};
} // ns cmm

#endif // CMM_TOOLS_SCOPE_HH_
