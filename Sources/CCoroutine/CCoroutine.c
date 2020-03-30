//
//  CCoroutine.c
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#include "CCoroutine.h"
#import <stdatomic.h>
#include <setjmp.h>

int __start(int* ret, const void* stack, const void* param, const void (*block)(const void*)) {
    int n = _setjmp(ret);
    if (n) return n;
    #if defined(__x86_64__)
    __asm__ ("movq %0, %%rsp" :: "g"(stack));
    block(param);
    #elif defined(__arm64__)
    __asm__ (
    "mov sp, %0\n"
    "mov x0, %1\n"
    "blr %2" :: "r"(stack), "r"(param), "r"(block));
    #endif
    return 0;
}

void __suspend(int* env, void** sp, int* ret, int retVal) {
    if (_setjmp(env)) return;
    char x; *sp = (void*)&x;
    _longjmp(ret, retVal);
}

int __save(int* env, int* ret, int retVal) {
    int n = _setjmp(ret);
    if (n) return n;
    _longjmp(env, retVal);
}

void __longjmp(int* env, int retVal) {
    _longjmp(env, retVal);
}

long __atomicExchange(_Atomic long* value, long desired) {
    return atomic_exchange(value, desired);
}

void __atomicStore(_Atomic long* value, long desired) {
    atomic_store(value, desired);
}
