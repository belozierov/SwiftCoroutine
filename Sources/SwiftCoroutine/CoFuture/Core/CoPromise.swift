//
//  CoPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

/// A promise to provide a result later.
///
/// `CoPromise` is subclass of `CoFuture`, що має методи, які дозволяють fulfill it. Це дозволяє інкапсулювати result provider.
/// Ви можете тільки один раз засетати результат в `CoPromise`, всі інші рази будуть ігноруватись.
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
        future.whenComplete(setResult)
    }
    
}

extension CoPromise where Value == Void {
    
    @inlinable public func success() {
        setResult(.success(()))
    }
    
}
