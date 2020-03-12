//
//  CoFuture+always.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - always
    
    @inlinable
    public func always(_ callback: @escaping (Result<Value, Error>) -> Void) -> CoFuture {
        whenComplete(callback)
        return self
    }
    
    @inlinable
    public func alwaysWhenSuccess(_ callback: @escaping (Value) -> Void) -> CoFuture {
        whenSuccess(callback)
        return self
    }
    
    @inlinable
    public func alwaysWhenFailure(_ callback: @escaping (Error) -> Void) -> CoFuture {
        whenFailure(callback)
        return self
    }
    
}

