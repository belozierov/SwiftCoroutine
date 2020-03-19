//
//  CoFuture+whenComplete.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - always
    
    /// Adds an observer callback that is called when the `CoFuture` has any result.
    /// - Parameter callback: The callback that is called when the `CoFuture` is fulfilled.
    /// - returns: The current `CoFuture`.
    @inlinable public func always(_ callback: @escaping (Result<Value, Error>) -> Void) -> CoFuture {
        whenComplete(callback)
        return self
    }
    
    // MARK: - whenComplete
    
    /// Adds an observer callback that is called when the `CoFuture` has any result.
    /// - Parameter callback: The callback that is called when the `CoFuture` is fulfilled.
    public func whenComplete(_ callback: @escaping (Result<Value, Error>) -> Void) {
        lock()
        if let result = _result {
            unlock()
            callback(result)
        } else {
            append(callback: callback)
            unlock()
        }
    }
    
    /// Adds an observer callback that is called when the `CoFuture` has a success result.
    /// - Parameter callback: The callback that is called with the successful result of the `CoFuture`.
    @inlinable public func whenSuccess(_ callback: @escaping (Value) -> Void) {
        whenComplete { result in
            if case .success(let value) = result { callback(value) }
        }
    }
    
    /// Adds an observer callback is called when the `CoFuture` has a failure result.
    /// - Parameter callback: The callback that is called with the failed result of the `CoFuture`.
    @inlinable public func whenFailure(_ callback: @escaping (Error) -> Void) {
        whenComplete { result in
            if case .failure(let error) = result { callback(error) }
        }
    }
    
    /// Adds an observer callback is called when the `CoFuture` was canceled.
    /// - Parameter callback: The callback that is called when the `CoFuture` was canceled.
    @inlinable public func whenCanceled(_ callback: @escaping () -> Void) {
        whenComplete { result in
            if case .failure(let error as CoFutureError) = result,
                error == .canceled { callback() }
        }
    }
    
}
