//
//  CoFututre+Operators.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    @inlinable public func map<T>(_ transformer: @escaping (Output) throws -> T) -> CoFuture<T> {
        transform { result in try transformer(result.get()) }
    }

     @inlinable public func then(_ handler: @escaping (Output) throws -> Void) -> CoFuture<Output> {
        map { value in
            try? handler(value)
            return value
        }
    }

    @inlinable public func `catch`(_ handler: @escaping (Error) throws -> Void) -> CoFuture<Output> {
        transform { result in
            switch result {
            case .success(let value):
                return value
            case .failure(let error):
                try? handler(error)
                throw error
            }
        }
    }
    
}
