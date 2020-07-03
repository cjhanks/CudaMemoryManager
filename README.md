# Cuda My Misery

*And fwiw - I think the CUDA team has done a great job.*

This library is being developed as the basis for an [audio processing library](https://github.com/cjhanks/cmm-audio),
but it is also a great opportunity to learn the less treaded path.

## Quickstart

```
$> cmake .. && make -j8
```

## Problem 1:  Memory Management

CUDA memory management and synchronization is difficult to get right.

NVIDIA has provided Uniform Memory between the GPU and the CPU.  On Intel chips
this (appears) is implemented as a page fault handler.  The default transfer
logic makes the memory unsuitable for performance critical applications.  Once
you tune the memory transfer logic, you have written more code than would have
been necessary if you avoided uniform memory altogether.  On some ARM64
platforms the physical memory is shared between the host and the client, so
there is no need for transfer logic in either case.

The only way to write an application which is performant on all target platforms
is to avoid any automated memory logic.  But this still leaves a few obstacles
which complicate application development.

1.  It is difficult to free temporary memory without forcing a synchronization
    of the program.  The synchronization point becomes a maintenance burden that
    shifts as a function of changes in program flow.
2.  When migrating data between OS Threads and/or CUDA Streams, it can be
    difficult to ensure data has been appropriately synchronized.
3.  Allocations pause the world, so it is desirable to avoid them.


## Problem 2:  CUDA Inter-Process Memory Management is Too Difficult

CUDA theoretically supports IPC memory.  It requires a lot of synchronization
tooling and C++ logic to make it possible and a lot of time to make it correct.

Separating different applications into different process spaces absolutely has
advantages.  A lot of people avoid doing it because making it performant is a
challenge greater than the perceived benefit.


## Problem 3:  CUDA *Still* Ignores C++ as the first class citizen

The majority of major real-world scientific products are not written in C, they
are written in C++.  For better or worse, that is where we are.

NVIDIA-Thrust was the solution that was offered.  And it was a pretty great
solution, for its time.  Scientific programming has simultaneously continued to
push the boundaries of performance while seeing a general decrease in software
engineering competency.  This is not a bad thing, an increasing number of
developers care about performance.

Those developers need abstractions that are more in line with their existing
logical models of programming.

- No weird macros.
- Asynchronous code should be programmed as if it were synchronous.
- Flow control between the CPU thread and the GPU stream should be synchronous.
- Error checking should not be so arduous.

That is an opinionated model of computing, I understand.
