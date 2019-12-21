//
//  CoFututre+Await.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension CoFuture {
    
    @inline(__always) public func await() throws -> Output {
        try await(coroutine: try currentCoroutine())
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
            isTimeOut ? .failure(FutureError.timeout) : nil
        }, resume: {
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
                       resultGetter: () -> OutputResult? = { nil },
                       resume: @escaping () -> Bool = { true },
                       suspendCompletion: (() -> Void)? = nil) throws -> Output {
        mutex.lock()
        defer { mutex.unlock() }
        defer { setCompletion(for: coroutine, completion: nil) }
        while true {
            if let result = result ?? resultGetter() {
                return try result.get()
            }
            setCompletion(for: coroutine) { _ in
                if resume() { coroutine.resume() }
            }
            coroutine.suspend {
                self.mutex.unlock()
                suspendCompletion?()
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
