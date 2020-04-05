//
//  PsxLock.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 02.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if os(Linux)
import Glibc
#else
import Darwin
#endif

@usableFromInline internal struct PsxLock {
    
    @usableFromInline internal let mutex: UnsafeMutablePointer<pthread_mutex_t>
    
    @inlinable internal init() {
        mutex = .allocate(capacity: 1)
        pthread_mutex_init(mutex, nil)
    }
    
    @inlinable internal func lock() {
        pthread_mutex_lock(mutex)
    }
    
    @inlinable internal func unlock() {
        pthread_mutex_unlock(mutex)
    }
    
    @inlinable internal func free() {
        pthread_mutex_destroy(mutex)
        mutex.deallocate()
    }
    
}
