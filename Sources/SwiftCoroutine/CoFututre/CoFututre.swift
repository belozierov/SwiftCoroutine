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
    
    // MARK: - Subscriptions
    
    private var subscriptions = [AnyHashable: Completion]()
    
    func setSubscription(for key: AnyHashable, completion: Completion?) {
        mutex.lock()
        subscriptions[key] = completion
        mutex.unlock()
    }
    
    func _addSubscription(completion: @escaping Completion) {
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
    
    @inline(__always)
    open func transform<T>(_ transformer: @escaping (Result) throws -> T) -> CoFuture<T> {
        CoTransformFuture(parent: self, transformer: transformer)
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
