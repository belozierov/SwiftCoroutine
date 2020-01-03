//
//  CoLazyPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

public class CoLazyPromise<Output>: CoFuture<Output> {
    
    public typealias PromiseBlock = (@escaping OutputHandler) -> Void
    
    private let promise: PromiseBlock
    private var _result: OutputResult?
    private var started = false
    
    public init(promise: @escaping PromiseBlock) {
        self.promise = promise
    }
    
    public init(on dispatcher: Dispatcher, promise: @escaping PromiseBlock) {
        self.promise = { completion in dispatcher.perform { promise(completion) } }
    }
    
    public convenience init(on dispatcher: Dispatcher, block: @escaping () throws -> Output) {
        self.init(on: dispatcher) { $0(Result(catching: block)) }
    }
    
    public override var result: OutputResult? {
        mutex.lock()
        if !started {
            started = true
            mutex.unlock()
            promise(complete)
            mutex.lock()
            defer { mutex.unlock() }
            return _result
        }
        defer { mutex.unlock() }
        return _result
    }
    
    override func saveResult(_ result: OutputResult) {
        _result = result
    }
    
}
