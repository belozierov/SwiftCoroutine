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
    
    public enum FutureError: Swift.Error {
        case timeOut, cancelled
    }
    
    let mutex = NSLock()
    var _result: Result?
    var subscriptions = [AnyHashable: Completion]()
    
    private func addSubscription(completion: @escaping Completion) {
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
    
    // MARK: - Operations
    
    open func await() throws -> Output {
        guard let coroutine = Coroutine.current
            else { fatalError() }
        mutex.lock()
        defer { mutex.unlock() }
        while true {
            if let result = _result { return try result.get() }
            addSubscription { _ in coroutine.resume() }
            mutex.unlock()
            coroutine.suspend()
            mutex.lock()
        }
    }
    
    @inline(__always) open func cancel() {
        finish(with: .failure(FutureError.cancelled))
    }
    
    // MARK: - Finish
    
    func finish(with result: Result) {
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
        subscriptions[promise] = { promise.finish(with: $0.map(transform)) }
        return promise
    }
    
    // MARK: - Notify
    
    open func notify(flags: DispatchWorkItemFlags = [], queue: DispatchQueue, execute: @escaping Completion) {
        mutex.lock()
        addSubscription { result in
            queue.async(flags: flags) { execute(result) }
        }
        mutex.unlock()
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
