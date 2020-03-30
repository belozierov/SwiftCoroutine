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

@usableFromInline struct PsxLock {
    
    @usableFromInline let mutex: UnsafeMutablePointer<pthread_mutex_t>
    
    @inlinable init() {
        mutex = .allocate(capacity: 1)
        pthread_mutex_init(mutex, nil)
    }
    
    @inlinable func lock() {
        pthread_mutex_lock(mutex)
    }
    
    @inlinable func unlock() {
        pthread_mutex_unlock(mutex)
    }
    
    @inlinable func free() {
        pthread_mutex_destroy(mutex)
        mutex.deallocate()
    }
    
}
