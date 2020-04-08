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

@usableFromInline protocol CoroutineProtocol: class {
    
    typealias StackSize = Coroutine.StackSize
    
    func await<T>(_ callback: (@escaping (T) -> Void) -> Void) -> T
    func await<T>(on scheduler: CoroutineScheduler, task: () throws -> T) rethrows -> T
    
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
    
    @inlinable internal static var current: CoroutineProtocol {
        guard let pointer = currentPointer else {
            precondition(false,
                         """
            Await must be called inside a coroutine.
            
            To launch the coroutine, use `startCoroutine()`, e.g. `DispatchQueue.main.startCoroutine()`.
            OR
            To check if inside the coroutine, use `Coroutine.isInsideCoroutine`.
            """)
            return PseudoCoroutine.shared
        }
        return Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue() as! CoroutineProtocol
    }
    
}

extension pthread_key_t {
    
    @usableFromInline internal static let coroutine: pthread_key_t = {
        var key: pthread_key_t = .zero
        pthread_key_create(&key, nil)
        return key
    }()
    
}


