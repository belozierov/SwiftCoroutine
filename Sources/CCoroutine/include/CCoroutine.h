//
//  CCoroutine.h
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#ifndef CCoroutine_h
#define CCoroutine_h

int __start(void* ret, const void* stack, const void* param, const void (*block)(const void*));
void __suspend(void* env, void** sp, void* ret, int retVal);
int __save(void* env, void* ret, int retVal);
void __longjmp(void* env, int retVal);

long __atomicExchange(_Atomic long* value, long desired);
void __atomicStore(_Atomic long* value, long desired);

#endif
