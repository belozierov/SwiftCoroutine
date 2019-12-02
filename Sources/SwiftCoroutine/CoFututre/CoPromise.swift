//
//  CoPromise.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 26.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

open class CoPromise<Output>: CoFuture<Output> {
    
    @inline(__always) public override init() {}
    
    @inline(__always) open func send(_ value: Output) {
        finish(with: .success(value))
    }
    
    @inline(__always) open func send(error: Error) {
        finish(with: .failure(error))
    }
    
    @inline(__always) open func send(result: Result) {
        finish(with: result)
    }
    
    @inline(__always) open func perform(_ block: () throws -> Output) {
        finish(with: Result(catching: block))
    }
    
}
