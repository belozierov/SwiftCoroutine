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
        let unmanaged = Unmanaged.passRetained(self)
        defer { unmanaged.release() }
        if let caller = pthread_getspecific(.coroutine) {
            pthread_setspecific(.coroutine, unmanaged.toOpaque())
            defer { pthread_setspecific(.coroutine, caller) }
            return block()
        } else {
            pthread_setspecific(.coroutine, unmanaged.toOpaque())
            defer { pthread_setspecific(.coroutine, nil) }
            return block()
        }
    }
    
}

extension Coroutine {
    
    /// Returns `true` if this property is called inside a coroutine.
    @inlinable public static var isInsideCoroutine: Bool {
        pthread_getspecific(.coroutine) != nil
    }
    
    @inlinable static func current() throws -> CoroutineProtocol {
        guard let pointer = pthread_getspecific(.coroutine)
            else { throw CoroutineError.mustBeCalledInsideCoroutine }
        return Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue() as! CoroutineProtocol
    }
    
}

extension pthread_key_t {
    
    @usableFromInline internal static let coroutine: pthread_key_t = {
        let key = UnsafeMutablePointer<pthread_key_t>.allocate(capacity: 1)
        pthread_key_create(key, nil)
        defer { key.deallocate() }
        return key.pointee
    }()
    
}


