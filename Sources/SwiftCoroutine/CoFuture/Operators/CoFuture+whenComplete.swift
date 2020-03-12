//
//  CoFuture+whenComplete.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - whenComplete
    
    @inlinable public func whenComplete(_ callback: @escaping (Result<Value, Error>) -> Void) {
        _whenComplete(callback: callback)
    }
    
    @usableFromInline func _whenComplete(callback: @escaping (Result<Value, Error>) -> Void) {
        lock()
        if let result = _result {
            unlock()
            callback(result)
        } else {
            append(callback: callback)
            unlock()
        }
    }
    
    @inlinable public func whenSuccess(_ callback: @escaping (Value) -> Void) {
        whenComplete { result in
            if case .success(let value) = result { callback(value) }
        }
    }
    
    @inlinable public func whenFailure(_ callback: @escaping (Error) -> Void) {
        whenComplete { result in
            if case .failure(let error) = result { callback(error) }
        }
    }
    
    @inlinable public func whenCancelled(_ callback: @escaping () -> Void) {
        _whenComplete { result in
            if case .failure(let error as CoFutureError) = result,
                error == .cancelled { callback() }
        }
    }
    
}
