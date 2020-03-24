//
//  NIOAsync.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import class NIO.EventLoopFuture
import protocol NIO.EventLoop
import SwiftCoroutine

extension EventLoop {
    
    public func async<Result>(_ closure: @escaping () throws -> Result) -> EventLoopFuture<Result> {
        let promise = makePromise(of: Result.self)
        let scheduler = TaskScheduler(scheduler: execute) { self.inEventLoop }
        CoroutineTaskExecutor.defaultShared.execute(on: scheduler) {
            promise.completeWith(.init(catching: closure))
        }
        return promise.futureResult
    }
}

extension EventLoopFuture {

    @inlinable public func await() throws -> Value {
        try Coroutine.await(whenComplete).get()
    }
    
}

