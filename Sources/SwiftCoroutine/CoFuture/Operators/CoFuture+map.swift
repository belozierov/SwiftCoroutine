//
//  CoFuture+map.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - map
    
    @inlinable public func map<NewValue>(_ transform: @escaping (Value) throws -> NewValue) -> CoFuture<NewValue> {
        mapResult { result in
            Result { try transform(result.get()) }
        }
    }
    
    @inlinable public func recover(_ transform: @escaping (Error) throws -> Value) -> CoFuture {
        mapResult { result in
            result.flatMapError { error in
                Result { try transform(error) }
            }
        }
    }
    
    public func mapResult<NewValue>(_ transform: @escaping (Result<Value, Error>) -> Result<NewValue, Error>) -> CoFuture<NewValue> {
        lock()
        if let result = _result {
            unlock()
            return CoFuture<NewValue>(result: transform(result))
        }
        let promise = CoPromise<NewValue>()
        addChild(future: promise) { result in
            promise.setResult(transform(result))
        }
        unlock()
        return promise
    }
    
}
