//
//  ThreadCoroutineWrapper.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Darwin

@usableFromInline final class ThreadCoroutineWrapper {
    
    @usableFromInline var coroutine: CoroutineProtocol?
    
    @usableFromInline static var current: ThreadCoroutineWrapper {
        if let pointer = pthread_getspecific(key) {
            return Unmanaged<ThreadCoroutineWrapper>.fromOpaque(pointer).takeUnretainedValue()
        }
        let wrapper = ThreadCoroutineWrapper()
        pthread_setspecific(key, Unmanaged.passRetained(wrapper).toOpaque())
        return wrapper
    }

    private static let key: pthread_key_t = {
        let key = UnsafeMutablePointer<pthread_key_t>.allocate(capacity: 1)
        pthread_key_create(key, nil)
        defer { key.deallocate() }
        return key.pointee
    }()
    
}
