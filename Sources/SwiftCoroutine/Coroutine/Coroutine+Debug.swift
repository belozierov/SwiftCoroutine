//
//  Coroutine+Debug.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 19.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension Coroutine {
    
    @inline(never) public static var currentStackUsed: Int? {
        Coroutine.current?.distanceToStack(from: _frameAddress())
    }
    
    @inline(never) public static var currentStackFreeSpace: Int? {
        guard let coroutine = Coroutine.current else { return nil }
        return coroutine.stackSize - coroutine.distanceToStack(from: _frameAddress())
    }
    
    @inline(__always) open var stackSize: Int {
        context.stackSize
    }
    
    private func distanceToStack(from pointer: UnsafeRawPointer) -> Int {
        pointer.distance(to: context.stackStart)
    }
    
}
