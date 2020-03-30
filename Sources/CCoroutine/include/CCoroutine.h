//
//  CCoroutine.h
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#ifndef CCoroutine_h
#define CCoroutine_h

int __start(int* ret, const void* stack, const void* param, const void (*block)(const void*));
void __suspend(int* env, void** sp, int* ret, int retVal);
int __save(int* env, int* ret, int retVal);
void __longjmp(int* env, int retVal);

long __atomicExchange(_Atomic long* value, long desired);
void __atomicStore(_Atomic long* value, long desired);

#endif
