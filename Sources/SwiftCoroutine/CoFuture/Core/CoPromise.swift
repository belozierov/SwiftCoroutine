//
//  CoPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

/// A promise to provide a result later.
///
/// `CoPromise` is a subclass of `CoFuture` that allows to deliver the result.
/// You can set the result to `CoPromise` only once, other attempts will be ignored.
///
public final class CoPromise<Value>: CoFuture<Value> {}

extension CoPromise {
    
    public convenience init() {
        self.init(_result: nil)
    }
    
    @inlinable public func complete<E: Error>(with result: Result<Value, E>) {
        switch result {
        case .success(let value): setResult(.success(value))
        case .failure(let error): setResult(.failure(error))
        }
    }
    
    @inlinable public func success(_ value: Value) {
        setResult(.success(value))
    }
    
    @inlinable public func fail(_ error: Error) {
        setResult(.failure(error))
    }
    
    @inlinable public func complete(with future: CoFuture<Value>) {
        future.addCallback(setResult)
    }
    
}
