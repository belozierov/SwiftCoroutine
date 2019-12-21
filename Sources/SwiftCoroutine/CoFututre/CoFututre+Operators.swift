//
//  CoFututre+Operators.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension CoFuture {
    
    // MARK: - Transform
    
    @inlinable public func map<T>(_ transformer: @escaping (Output) throws -> T) -> CoFuture<T> {
        transform { result in try transformer(result.get()) }
    }

     @inlinable public func then(_ handler: @escaping (Output) -> Void) -> CoFuture<Output> {
        map { value in
            handler(value)
            return value
        }
    }

    @inlinable public func `catch`(_ handler: @escaping (Error) -> Void) -> CoFuture<Output> {
        transform { result in
            switch result {
            case .success(let value):
                return value
            case .failure(let error):
                handler(error)
                throw error
            }
        }
    }
    
    @inlinable public func handler(_ handler: @escaping () -> Void) -> CoFuture<Output> {
        transform { result in
            handler()
            return try result.get()
        }
    }
    
    // MARK: - Notify
    
    public func notify(execute completion: @escaping Completion) {
        mutex.lock()
        if let result = result {
            mutex.unlock()
            return completion(result)
        }
        addCompletion(completion: completion)
        mutex.unlock()
    }
    
    @inlinable public func notify(flags: DispatchWorkItemFlags = [], queue: DispatchQueue, execute completion: @escaping Completion) {
        notify { result in queue.async(flags: flags) { completion(result) } }
    }
    
}
