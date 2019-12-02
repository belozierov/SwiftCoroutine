//
//  CCoroutine.h
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#ifndef CCoroutine_h
#define CCoroutine_h

void __start(int* ret, const void* stack, const void* param, const void (*block)(const void*));
void __save(int* env, int* ret);

#endif
