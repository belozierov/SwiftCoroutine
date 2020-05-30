//
//  CoFuture3+flatMap.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - flatMap
    
    /// When the current `CoFuture` is fulfilled, run the provided callback, which will provide a new `CoFuture`.
    ///
    /// This allows you to dynamically dispatch new asynchronous tasks as phases in a
    /// longer series of processing steps. Note that you can use the results of the
    /// current `CoFuture` when determining how to dispatch the next operation.
    /// - Parameter callback: Function that will receive the value and return a new `CoFuture`.
    /// - returns: A future that will receive the eventual value.
    @inlinable public func flatMap<NewValue>(_ callback: @escaping (Value) -> CoFuture<NewValue>) -> CoFuture<NewValue> {
        flatMapResult { result in
            switch result {
            case .success(let value):
                return callback(value)
            case .failure(let error):
                return CoFuture<NewValue>(result: .failure(error))
            }
        }
    }
    
    /// When the current `CoFuture` is in an error state, run the provided callback, which
    /// may recover from the error by returning a `CoFuture`.
    /// - Parameter callback: Function that will receive the error value and return a new value lifted into a new `CoFuture`.
    /// - returns: A future that will receive the recovered value.
    @inlinable public func flatMapError(_ callback: @escaping (Error) -> CoFuture) -> CoFuture {
        flatMapResult { result in
            switch result {
            case .success:
                return CoFuture(result: result)
            case .failure(let error):
                return callback(error)
            }
        }
    }
    
    /// When the current `CoFuture` is fulfilled, run the provided callback, which will provide a new `CoFuture`.
    /// - Parameter callback: Function that will receive the result and return a new `CoFuture`.
    /// - returns: A future that will receive the eventual value.
    public func flatMapResult<NewValue>(_ callback: @escaping (Result<Value, Error>) -> CoFuture<NewValue>) -> CoFuture<NewValue> {
        if let result = result {
            return callback(result)
        }
        let promise = CoPromise<NewValue>(parent: self)
        addCallback { callback($0).addCallback(promise.setResult) }
        return promise
    }
    
}
