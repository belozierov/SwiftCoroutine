//
//  CoFututre+Operators.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    public typealias Dispatcher = Coroutine.Dispatcher
    
    // MARK: - Transform
    
    public func transform<T>(_ transformer: @escaping (OutputResult) throws -> T) -> CoFuture<T> {
        CoTransformFuture(parent: self, transformer: transformer)
    }
    
    // MARK: - Map output
    
    @inlinable public func transformValue<T>(_ transformer: @escaping (Output) throws -> T) -> CoFuture<T> {
        transform { result in try transformer(result.get()) }
    }
    
    // MARK: - On result
    
    @discardableResult
    public func onResult(on dispatcher: Dispatcher = .sync, execute completion: @escaping OutputHandler) -> CoFuture<Output> {
        CoHandleFuture(parent: self) { result in
            dispatcher.dispatchBlock { completion(result) }
        }
    }
    
    @inlinable @discardableResult
    public func onResult(on dispatcher: Dispatcher = .sync, execute completion: @escaping () -> Void) -> CoFuture<Output> {
        onResult(on: dispatcher) { _ in completion() }
    }
    
    // MARK: - On success
    
    @inlinable @discardableResult
    public func onSuccess(on dispatcher: Dispatcher = .sync, execute handler: @escaping (Output) -> Void) -> CoFuture<Output> {
        onResult(on: dispatcher) {
            if case .success(let output) = $0 { handler(output) }
        }
    }
    
    // MARK: - On error
    
    @inlinable @discardableResult
    public func onError(on dispatcher: Dispatcher = .sync, execute handler: @escaping (Error) -> Void) -> CoFuture<Output> {
        onResult(on: dispatcher) {
            if case .failure(let error) = $0 { handler(error) }
        }
    }
    
}
