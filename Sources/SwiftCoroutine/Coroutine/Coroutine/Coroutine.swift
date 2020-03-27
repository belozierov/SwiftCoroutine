//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 01.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

public struct Coroutine {
    
    // MARK: - current
    
    @inlinable static func current() throws -> CoroutineProtocol {
        if let coroutine = ThreadCoroutineWrapper.current.coroutine { return coroutine }
        throw CoroutineError.mustBeCalledInsideCoroutine
    }
    
    @inlinable public static var isInsideCoroutine: Bool {
        ThreadCoroutineWrapper.current.coroutine != nil
    }
    
    // MARK: - await
    
    @inlinable public static func await(_ callback: (@escaping () -> Void) -> Void) throws {
        try await { callback($0) }
    }
    
    @inlinable public static func await<T>(_ callback: (@escaping (T) -> Void) -> Void) throws -> T {
        try current().await(callback)
    }
    
    @inlinable public static func await<T, N>(_ callback: (@escaping (T, N) -> Void) -> Void) throws -> (T, N) {
        try current().await { completion in callback { a, b in completion((a, b)) } }
    }
    
    @inlinable public static func await<T, N, M>(_ callback: (@escaping (T, N, M) -> Void) -> Void) throws -> (T, N, M) {
        try current().await { completion in callback { a, b, c in completion((a, b, c)) } }
    }
    
    // MARK: - delay
    
    @inlinable public static func delay(_ time: DispatchTime) throws {
        try await { DispatchSource.createTimer(timeout: time, handler: $0).activate() }
    }
    
}
