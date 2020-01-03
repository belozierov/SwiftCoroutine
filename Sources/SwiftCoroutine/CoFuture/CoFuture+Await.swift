//
//  CoFuture+Await.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Dispatch

extension CoFuture {
    
    public func await() throws -> Output {
        let coroutine = try Coroutine.current()
        while true {
            if let result = result { return try result.get() }
            coroutine.suspend {
                self.subscribe(with: coroutine, handler: coroutine.resume)
            }
        }
    }
    
    public func await(timeout: DispatchTime) throws -> Output {
        let coroutine = try Coroutine.current(), mutex = self.mutex
        var isTimeOut = false
        let timer = DispatchSource.startTimer(timeout: timeout) {
            mutex.lock()
            isTimeOut = true
            mutex.unlock()
            if coroutine.state == .suspended { coroutine.resume() }
        }
        defer { timer.cancel() }
        while true {
            mutex.lock()
            if isTimeOut { throw FutureError.timeout }
            if let result = result { return try result.get() }
            coroutine.suspend {
                self.subscribe(with: coroutine) {
                    if coroutine.state == .suspended { coroutine.resume() }
                }
                mutex.unlock()
            }
        }
    }
    
}

extension DispatchSource {
    
    fileprivate static func startTimer(timeout: DispatchTime, handler: @escaping () -> Void) -> DispatchSourceTimer {
        let timer = DispatchSource.makeTimerSource()
        timer.schedule(deadline: timeout, leeway: .milliseconds(50))
        timer.setEventHandler(handler: handler)
        timer.activate()
        return timer
    }
    
}
