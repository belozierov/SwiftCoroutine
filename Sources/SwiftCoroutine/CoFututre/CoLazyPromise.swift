//
//  CoLazyPromise.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 13.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class CoLazyPromise<Output>: CoFuture<Output> {
    
    public typealias Completion = (Result) -> Void
    public typealias PromiseBlock = (@escaping Completion) -> Void
    
    private let promise: PromiseBlock
    private var __result: Result?
    private var started = false
    
    public init(promise: @escaping PromiseBlock) {
        self.promise = promise
    }
    
    public init(queue: DispatchQueue, promise: @escaping PromiseBlock) {
        self.promise = { completion in queue.async { promise(completion) } }
    }
    
    override var _result: Result? {
        set { __result = newValue }
        get {
            if let result = __result { return result }
            if !started { start() }
            return __result
        }
    }
    
    private func start() {
        started = true
        mutex.unlock()
        promise(finish)
        mutex.lock()
    }
    
}
