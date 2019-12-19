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

int __start(jmp_buf ret, const void* stack, const void* param, const void (*block)(const void*));
int __save(jmp_buf env, jmp_buf ret, int retVal);

const void* _frameAddress(void);

#endif
