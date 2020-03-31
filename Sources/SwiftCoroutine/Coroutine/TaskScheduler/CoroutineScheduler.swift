//
//  CoroutineScheduler.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public protocol CoroutineScheduler {
    
    func scheduleTask(_ task: @escaping () -> Void)
    
}

extension CoroutineScheduler {
    
    // MARK: - coroutine
    
    @inlinable public func startCoroutine(_ task: @escaping () -> Void) {
        CoroutineDispatcher.default.execute(on: self, task: task)
    }
    
    @inlinable public func startCoroutine(_ task: @escaping () throws -> Void) {
        startCoroutine { do { try task() } catch { print(error) } }
    }
    
    // MARK: - await
    
    @inlinable public func await<T>(_ task: @escaping () throws -> T) throws -> T {
        try Coroutine.await { callback in
            startCoroutine { callback(Result(catching: task)) }
        }.get()
    }
    
    // MARK: - future
    
    @inlinable public func coFuture<T>(_ task: @escaping () throws -> T) -> CoFuture<T> {
        let promise = CoPromise<T>()
        startCoroutine { promise.complete(with: Result(catching: task)) }
        return promise
    }
    
}
