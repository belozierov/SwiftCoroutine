//
//  CCoroutine.c
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#include "CCoroutine.h"
#include <setjmp.h>

void __start(jmp_buf ret, const void* stack, const void* param, const void (*block)(const void*)) {
    if (setjmp(ret)) return;
    #if defined(__x86_64__)
    __asm__ ("movq %0, %%rsp" :: "g"(stack));
    block(param);
    #elif defined(__arm64__)
    __asm__ (
    "mov sp, %0\n"
    "mov x0, %1\n"
    "blr %2\n"
    :: "r"(stack), "r"(param), "r"(block));
    #endif
}

void __save(jmp_buf env, jmp_buf ret) {
    if (!setjmp(env)) longjmp(ret, 1);
}
