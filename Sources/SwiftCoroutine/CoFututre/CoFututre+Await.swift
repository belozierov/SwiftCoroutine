//
//  CoFututre+Await.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension CoFuture {
    
    @inline(__always) public func await() throws -> Output {
        try await(coroutine: try currentCoroutine(), resultGetter: { _result })
    }
    
    public func await(timeout: DispatchTime) throws -> Output {
        let coroutine = try currentCoroutine()
        let mutex = NSLock()
        var isSuspended = false, isTimeOut = false
        let timer = DispatchSource.makeTimerSource()
        timer.schedule(deadline: timeout, leeway: .milliseconds(100))
        timer.setEventHandler {
            mutex.lock()
            isTimeOut = true
            if !isSuspended { return mutex.unlock() }
            isSuspended = false
            mutex.unlock()
            coroutine.resume()
        }
        timer.activate()
        defer { timer.cancel() }
        return try await(coroutine: coroutine, resultGetter: {
            _result ?? (isTimeOut ? .failure(FutureError.timeout) : nil)
        }, resumeSubscription: {
            mutex.lock()
            defer { mutex.unlock() }
            if isTimeOut { return false }
            defer { isSuspended = false }
            return isSuspended
        }, suspendCompletion: {
            mutex.lock()
            isSuspended = true
            mutex.unlock()
        })
    }
    
    private func await(coroutine: Coroutine,
                       resultGetter: () -> Result?,
                       resumeSubscription: @escaping () -> Bool = { true },
                       suspendCompletion: @escaping () -> Void = {}) throws -> Output {
        mutex.lock()
        defer { mutex.unlock() }
        while true {
            if let result = resultGetter() { return try result.get() }
            _addSubscription { _ in
                if resumeSubscription() { coroutine.resume() }
            }
            coroutine.suspend {
                self.mutex.unlock()
                suspendCompletion()
            }
            mutex.lock()
        }
    }
    
    private func currentCoroutine() throws -> Coroutine {
        assert(Coroutine.current != nil, "Await must be called inside coroutine")
        guard let coroutine = Coroutine.current
            else { throw FutureError.awaitCalledOutsideCoroutine }
        return coroutine
    }
    
}
