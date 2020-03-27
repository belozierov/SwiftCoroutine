//
//  CCoroutine.c
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#include "CCoroutine.h"
#import <stdatomic.h>

int __start(jmp_buf ret, const void* stack, const void* param, const void (*block)(const void*)) {
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

void __suspend(jmp_buf env, void** sp, jmp_buf ret, int retVal) {
    if (_setjmp(env)) return;
    char x; *sp = (void*)&x;
    _longjmp(ret, retVal);
}

int __save(jmp_buf env, jmp_buf ret, int retVal) {
    int n = _setjmp(ret);
    if (n) return n;
    _longjmp(env, retVal);
}

int __atomicCompareExchange(_Atomic long* value, long* expected, long desired) {
    return atomic_compare_exchange_weak(value, expected, desired);
}

long __atomicExchange(_Atomic long* value, long desired) {
    return atomic_exchange(value, desired);
}

void __atomicStore(_Atomic long* value, long desired) {
    atomic_store(value, desired);
}
