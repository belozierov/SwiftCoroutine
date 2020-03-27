//
//  TaskExecutor.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public protocol TaskExecutor {
    
    func execute(_ task: @escaping () -> Void)
    
}

extension TaskExecutor {
    
    @inlinable public func execute(_ task: @escaping () throws -> Void) {
        execute { do { try task() } catch { print(error) } }
    }
    
    // MARK: - await
    
    @inlinable public func await<T>(_ callback: @escaping (@escaping (T) -> Void) -> Void) throws -> T {
        try Coroutine.await { completion in execute { callback(completion) } }
    }
    
    @inlinable public func await(_ callback: @escaping (@escaping () -> Void) -> Void) throws {
        try Coroutine.await { completion in execute { callback(completion) } }
    }

    @inlinable public func await<T, N>(_ callback: @escaping (@escaping (T, N) -> Void) -> Void) throws -> (T, N) {
        try Coroutine.await { completion in execute { callback(completion) } }
    }

    @inlinable public func await<T, N, M>(_ callback: @escaping (@escaping (T, N, M) -> Void) -> Void) throws -> (T, N, M) {
        try Coroutine.await { completion in execute { callback(completion) } }
    }
    
    @inlinable public func await<T>(_ callback: @escaping () throws -> T) throws -> T {
        try await { $0(Result(catching: callback)) }.get()
    }
    
    // MARK: - CoFuture
    
    @inlinable public func submit<T>(_ task: @escaping () throws -> T) -> CoFuture<T> {
        let promise = CoPromise<T>()
        execute { promise.complete(with: Result { try task() }) }
        return promise
    }
    
    @inlinable public func submit<T>(_ task: @escaping () -> T) -> CoFuture<T> {
        CoFuture { $0.success(task()) }
    }
    
}

extension TaskScheduler: TaskExecutor {}
extension CoroutineDispatcher: TaskExecutor {}

