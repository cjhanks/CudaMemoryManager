# **C**uda **M**y **M**isery

## Quickstart

```
$> cmake .. && make -j8
```

## Problem

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

## A Solution


