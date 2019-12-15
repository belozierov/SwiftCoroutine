//
//  CoPromise.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 26.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

open class CoPromise<Output>: CoFuture<Output> {
    
    @inline(__always) public override init() {}
    
    @inlinable open func send(_ value: Output) {
        finish(with: .success(value))
    }
    
    @inlinable open func send(error: Error) {
        finish(with: .failure(error))
    }
    
    @inlinable open func send(result: Result) {
        finish(with: result)
    }
    
    @inlinable open func perform(_ block: () throws -> Output) {
        finish(with: Result(catching: block))
    }
    
}
