//
//  CoFuture+flatMap.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - flatMap
    
    @inlinable public func flatMap<NewValue>(_ transform: @escaping (Value) -> CoFuture<NewValue>) -> CoFuture<NewValue> {
        flatMapResult { result in
            switch result {
            case .success(let value):
                return transform(value)
            case .failure(let error):
                return CoFuture<NewValue>(result: .failure(error))
            }
        }
    }
    
    @inlinable public func flatMapError(_ transform: @escaping (Error) -> CoFuture) -> CoFuture {
        flatMapResult { result in
            switch result {
            case .success:
                return CoFuture(result: result)
            case .failure(let error):
                return transform(error)
            }
        }
    }
    
    public func flatMapResult<NewValue>(_ transform: @escaping(Result<Value, Error>) -> CoFuture<NewValue>) -> CoFuture<NewValue> {
        lock()
        if let result = _result {
            unlock()
            return transform(result)
        }
        let promise = CoPromise<NewValue>()
        addChild(future: promise) { result in
            transform(result).whenComplete(promise.setResult)
        }
        unlock()
        return promise
    }
    
}
