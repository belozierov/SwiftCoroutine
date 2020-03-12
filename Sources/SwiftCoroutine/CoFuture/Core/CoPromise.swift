//
//  CoPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public final class CoPromise<Value>: CoFuture<Value> {}

extension CoPromise {
    
    @inlinable public func complete(with result: Result<Value, Error>) {
        setResult(result)
    }
    
    @inlinable public func success(_ value: Value) {
        setResult(.success(value))
    }
    
    @inlinable public func fail(_ error: Error) {
        setResult(.failure(error))
    }
    
    @inlinable public func complete(with future: CoFuture<Value>) {
        future.whenComplete(complete)
    }
    
}
