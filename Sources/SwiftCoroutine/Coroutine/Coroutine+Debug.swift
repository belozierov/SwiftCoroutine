//
//  Coroutine+Debug.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 19.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif

extension Coroutine {
    
    @inline(never) public static var currentStackUsed: Int? {
        try? Coroutine.current().currentContext.usedStackSpace
    }
    
    @inline(never) public static var currentStackFreeSpace: Int? {
        try? Coroutine.current().currentContext.freeStackSpace
    }
    
    @inline(__always) public var stackSize: Int {
        currentContext.stackSize
    }
    
    private var currentContext: CoroutineContext {
        subRoutines.last?.context ?? context
    }
    
}

extension CoroutineContext {
    
    @inline(never) fileprivate var usedStackSpace: Int {
        _frameAddress().distance(to: stackStart)
    }
    
    fileprivate var freeStackSpace: Int {
        stackSize - usedStackSpace
    }
    
}
