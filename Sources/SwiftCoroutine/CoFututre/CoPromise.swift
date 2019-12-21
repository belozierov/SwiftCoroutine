//
//  CoPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

open class CoPromise<Output>: CoFuture<Output> {
    
    private var _result: OutputResult?
    
    public override init() {}
    
    open override var result: OutputResult? {
        mutex.lock()
        defer { mutex.unlock() }
        return _result
    }
    
    open override func send(result: OutputResult) {
        mutex.lock()
        _result = result
        mutex.unlock()
        super.send(result: result)
    }
    
}

extension CoPromise {
    
    @inlinable public func send(_ output: Output) {
        send(result: .success(output))
    }
    
    @inlinable public func send(error: Error) {
        send(result: .failure(error))
    }
    
    @inlinable public func perform(_ block: () throws -> Output) {
        send(result: Result(catching: block))
    }
    
}
