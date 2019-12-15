//
//  CoFututre.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 26.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class CoFuture<Output> {
    
    public typealias Result = Swift.Result<Output, Error>
    public typealias Completion = (Result) -> ()
    
    public enum FutureError: Error {
        case cancelled, awaitCalledOutsideCoroutine, timeout
    }
    
    let mutex = NSLock()
    var _result: Result?
    var subscriptions = [AnyHashable: Completion]()
    
    func addSubscription(completion: @escaping Completion) {
        withUnsafePointer(to: completion) { subscriptions[$0] = completion }
    }
    
    // MARK: - State
    
    open var result: Result? {
        mutex.lock()
        defer { mutex.unlock() }
        return _result
    }
    
    open var isCancelled: Bool {
        mutex.lock()
        defer { mutex.unlock() }
        if case .failure(let error as FutureError) = result {
            return error == .cancelled
        }
        return false
    }
    
    // MARK: - Await
    
    @inline(__always) open func await() throws -> Output {
        try await(coroutine: try currentCoroutine(), resultGetter: { _result })
    }
    
    open func await(timeout: DispatchTime) throws -> Output {
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
            addSubscription { _ in
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
    
    // MARK: - Finish
    
    @inlinable open func cancel() {
        finish(with: .failure(FutureError.cancelled))
    }
    
    @usableFromInline func finish(with result: Result) {
        mutex.lock()
        _result = result
        let blocks = subscriptions.values
        subscriptions.removeAll()
        mutex.unlock()
        blocks.forEach { $0(result) }
    }
    
    // MARK: - Transform
    
    open func transform<T>(_ transform: @escaping (Output) -> T) -> CoFuture<T> {
        let promise = CoPromise<T>()
        mutex.lock()
        subscriptions[promise] = { promise.finish(with: $0.map(transform)) }
        mutex.unlock()
        return promise
    }
    
    // MARK: - Notify
    
    open func notify(flags: DispatchWorkItemFlags = [], queue: DispatchQueue, execute: @escaping Completion) {
        mutex.lock()
        defer { mutex.unlock() }
        let completion = { result in queue.async(flags: flags) { execute(result) } }
        if let result = _result { return completion(result) }
        addSubscription(completion: completion)
    }
    
}

extension CoFuture: Hashable {
    
    public static func == (lhs: CoFuture, rhs: CoFuture) -> Bool {
        lhs === rhs
    }
    
    public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
