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
    
    @inline(never) public static func currentStackUsed() throws -> Int {
        try Coroutine.current().stackUsed(to: _frameAddress())
    }
    
    @inline(never) public static func currentStackFreeSpace() throws -> Int {
        try Coroutine.current().stackFreeSpace(to: _frameAddress())
    }
    
    private func stackUsed(to pointer: UnsafeRawPointer) -> Int {
        pointer.distance(to: currentContext.stackStart)
    }
    
    private func stackFreeSpace(to pointer: UnsafeRawPointer) -> Int {
        stackSize - stackUsed(to: pointer)
    }
    
    @inline(__always) public var stackSize: Int {
        currentContext.stackSize
    }
    
    private var currentContext: CoroutineContext {
        subRoutines.last?.context ?? context
    }
    
}


