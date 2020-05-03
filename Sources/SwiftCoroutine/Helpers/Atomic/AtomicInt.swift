//
//  AtomicInt.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 01.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif

@inlinable internal func atomicStore(_ pointer: UnsafeMutablePointer<Int>, value: Int) {
    __atomicStore(OpaquePointer(pointer), value)
}

@inlinable @discardableResult
internal func atomicAdd(_ pointer: UnsafeMutablePointer<Int>, value: Int) -> Int {
    __atomicFetchAdd(OpaquePointer(pointer), value)
}

@inlinable internal func atomicExchange(_ pointer: UnsafeMutablePointer<Int>, with value: Int) -> Int {
    __atomicExchange(OpaquePointer(pointer), value)
}

@discardableResult @inlinable
internal func atomicCAS(_ pointer: UnsafeMutablePointer<Int>, expected: Int, desired: Int) -> Bool {
    var expected = expected
    return __atomicCompareExchange(OpaquePointer(pointer), &expected, desired) != 0
}

@discardableResult @inlinable internal
func atomicUpdate(_ pointer: UnsafeMutablePointer<Int>, transform: (Int) -> Int) -> (old: Int, new: Int) {
    var oldValue = pointer.pointee, newValue: Int
    repeat { newValue = transform(oldValue) }
        while __atomicCompareExchange(OpaquePointer(pointer), &oldValue, newValue) == 0
    return (oldValue, newValue)
}
