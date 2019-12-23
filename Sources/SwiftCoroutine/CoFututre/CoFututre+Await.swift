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
        try await(coroutine: .current())
    }
    
    public func await(timeout: DispatchTime) throws -> Output {
        let coroutine = try Coroutine.current()
        let mutex = self.mutex
        var isSuspended = false, isTimeOut = false
        let timer = DispatchSource.startTimer(timeout: timeout) {
            mutex.lock()
            isTimeOut = true
            if !isSuspended { return mutex.unlock() }
            isSuspended = false
            mutex.unlock()
            coroutine.resume()
        }
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
            isSuspended = true
        })
    }
    
    private func await(coroutine: Coroutine,
                       resultGetter: () -> OutputResult? = { nil },
                       resume: @escaping () -> Bool = { true },
                       suspendCompletion: (() -> Void)? = nil) throws -> Output {
        mutex.lock()
        defer { mutex.unlock() }
        defer { completions[coroutine] = nil }
        while true {
            if let result = result ?? resultGetter() {
                return try result.get()
            }
            completions[coroutine] = { _ in
                if resume() { coroutine.resume() }
            }
            coroutine.suspend {
                suspendCompletion?()
                self.mutex.unlock()
            }
            mutex.lock()
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
