//
//  CCoroutine.h
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#ifndef CCoroutine_h
#define CCoroutine_h
#include <setjmp.h>

struct __CoroutineEnvironment {
    int* env; void* sp;
};

int __start(jmp_buf ret, const void* stack, const void* param, const void (*block)(const void*));
void __suspend(struct __CoroutineEnvironment* data, jmp_buf ret, int retVal);
int __save(jmp_buf env, jmp_buf ret, int retVal);

int __compare(_Atomic long* value, long* expected, long desired);

#endif
