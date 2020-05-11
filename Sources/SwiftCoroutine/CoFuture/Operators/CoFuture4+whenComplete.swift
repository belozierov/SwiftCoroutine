//
//  CoFuture2+whenComplete.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - whenComplete
    
    /// Adds an observer callback that is called when the `CoFuture` has any result.
    /// - Parameter callback: The callback that is called when the `CoFuture` is fulfilled.
    @inlinable public func whenComplete(_ callback: @escaping (Result<Value, Error>) -> Void) {
        addCallback(callback)
    }
    
    /// Adds an observer callback that is called when the `CoFuture` has a success result.
    /// - Parameter callback: The callback that is called with the successful result of the `CoFuture`.
    @inlinable public func whenSuccess(_ callback: @escaping (Value) -> Void) {
        addCallback { result in
            if case .success(let value) = result { callback(value) }
        }
    }
    
    /// Adds an observer callback that is called when the `CoFuture` has a failure result.
    /// - Parameter callback: The callback that is called with the failed result of the `CoFuture`.
    @inlinable public func whenFailure(_ callback: @escaping (Error) -> Void) {
        addCallback { result in
            if case .failure(let error) = result { callback(error) }
        }
    }
    
    /// Adds an observer callback that is called when the `CoFuture` is canceled.
    /// - Parameter callback: The callback that is called when the `CoFuture` is canceled.
    @inlinable public func whenCanceled(_ callback: @escaping () -> Void) {
        addCallback { result in
            if case .failure(let error as CoFutureError) = result,
                error == .canceled { callback() }
        }
    }
    
    /// Adds an observer callback that is called when the `CoFuture` has any result.
    /// - Parameter callback: The callback that is called when the `CoFuture` is fulfilled.
    @inlinable public func whenComplete(_ callback: @escaping () -> Void) {
        whenComplete { _ in callback() }
    }
    
}
