//
//  CoLazyPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class CoLazyPromise<Output>: CoFuture<Output> {
    
    public typealias PromiseBlock = (@escaping Completion) -> Void
    
    private let promise: PromiseBlock
    private var _result: OutputResult?
    private var started = false
    
    public init(promise: @escaping PromiseBlock) {
        self.promise = promise
    }
    
    public init(queue: DispatchQueue, promise: @escaping PromiseBlock) {
        self.promise = { completion in queue.async { promise(completion) } }
    }
    
    open override var result: OutputResult? {
        mutex.lock()
        defer { mutex.unlock() }
        if let result = _result { return result }
        if !started { start() }
        return _result
    }
    
    override func send(result: OutputResult) {
        mutex.lock()
        _result = result
        mutex.unlock()
        super.send(result: result)
    }
    
    private func start() {
        started = true
        mutex.unlock()
        promise(send)
        mutex.lock()
    }
    
}
