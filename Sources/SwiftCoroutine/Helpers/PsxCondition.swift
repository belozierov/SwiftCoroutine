//
//  PsxCondition.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 05.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if os(Linux)
import Glibc
#else
import Darwin
#endif

internal struct PsxCondition {
    
    internal let mutex = PsxLock()
    internal let condition: UnsafeMutablePointer<pthread_cond_t>
    
    @inlinable internal init() {
        condition = .allocate(capacity: 1)
        pthread_cond_init(condition, nil)
    }
    
    @inlinable internal func lock() {
        mutex.lock()
    }
    
    @inlinable internal func unlock() {
        mutex.unlock()
    }
    
    @inlinable internal func wait() {
        pthread_cond_wait(condition, mutex.mutex)
    }
    
    @inlinable internal func signal() {
        pthread_cond_signal(condition)
    }
    
    @inlinable internal func broadcast() {
        pthread_cond_broadcast(condition)
    }
    
    @inlinable internal func free() {
        pthread_cond_destroy(condition)
        condition.deallocate()
        mutex.free()
    }
    
}
