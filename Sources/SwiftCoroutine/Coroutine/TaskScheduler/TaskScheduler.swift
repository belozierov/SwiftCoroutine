//
//  TaskScheduler.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public protocol TaskScheduler {
    
    func executeTask(_ task: @escaping () -> Void)
    
}

extension TaskScheduler {
    
    // MARK: - coroutine
    
    @inlinable public func coroutine(_ task: @escaping () -> Void) {
        CoroutineDispatcher.default.execute(on: self, task: task)
    }
    
    @inlinable public func coroutine(_ task: @escaping () throws -> Void) {
        coroutine { do { try task() } catch { print(error) } }
    }
    
    // MARK: - await
    
    @inlinable public func await(_ callback: @escaping (@escaping () -> Void) -> Void) throws {
        try Coroutine.await { completion in executeTask { callback(completion) } }
    }
    
    @inlinable public func await<T>(_ callback: @escaping (@escaping (T) -> Void) -> Void) throws -> T {
        try Coroutine.await { completion in executeTask { callback(completion) } }
    }
    
    @inlinable public func await<T, N>(_ callback: @escaping (@escaping (T, N) -> Void) -> Void) throws -> (T, N) {
        try Coroutine.await { completion in executeTask { callback(completion) } }
    }

    @inlinable public func await<T, N, M>(_ callback: @escaping (@escaping (T, N, M) -> Void) -> Void) throws -> (T, N, M) {
        try Coroutine.await { completion in executeTask { callback(completion) } }
    }
    
    @inlinable public func await<T>(_ callback: @escaping () throws -> T) throws -> T {
        try await { $0(Result(catching: callback)) }.get()
    }
    
    // MARK: - future
    
    @inlinable public func coFuture<T>(_ task: @escaping () throws -> T) -> CoFuture<T> {
        let promise = CoPromise<T>()
        executeTask { promise.complete(with: Result(catching: task)) }
        return promise
    }
    
}
