//
//  CoroutineProtocol.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if os(Linux)
import Glibc
#else
import Darwin
#endif

@usableFromInline internal protocol CoroutineProtocol: AnyObject {
    
    typealias StackSize = Coroutine.StackSize
    
    func await<T>(_ callback: (@escaping (T) -> Void) -> Void) throws -> T
    func await<T>(on scheduler: CoroutineScheduler, task: () throws -> T) throws -> T
    func cancel()
    
}

extension CoroutineProtocol {
    
    @inlinable internal func performAsCurrent<T>(_ block: () -> T) -> T {
        let caller = pthread_getspecific(.coroutine)
        pthread_setspecific(.coroutine, Unmanaged.passUnretained(self).toOpaque())
        defer { pthread_setspecific(.coroutine, caller) }
        return block()
    }
    
}

extension Coroutine {
    
    @inlinable internal static var currentPointer: UnsafeMutableRawPointer? {
        pthread_getspecific(.coroutine)
    }
    
    @inlinable internal static func current() throws -> CoroutineProtocol {
        if let pointer = currentPointer,
            let coroutine = Unmanaged<AnyObject>.fromOpaque(pointer)
                .takeUnretainedValue() as? CoroutineProtocol {
            return coroutine
        }
        throw CoroutineError.calledOutsideCoroutine
    }
    
}

extension pthread_key_t {
    
    @usableFromInline internal static let coroutine: pthread_key_t = {
        var key: pthread_key_t = .zero
        pthread_key_create(&key, nil)
        return key
    }()
    
}


