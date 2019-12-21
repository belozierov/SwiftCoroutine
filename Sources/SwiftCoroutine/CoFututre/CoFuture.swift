//
//  CoFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class CoFuture<Output> {
    
    public typealias OutputResult = Result<Output, Error>
    public typealias Completion = (OutputResult) -> Void
    
    public enum FutureError: Error {
        case cancelled, awaitCalledOutsideCoroutine, timeout
    }
    
    let mutex = NSRecursiveLock()
    private var completions = [AnyHashable: Completion]()
    
    // MARK: - Result
    
    open var result: OutputResult? { nil }
    
    @usableFromInline func send(result: OutputResult) {
        mutex.lock()
        let items = completions.values
        completions.removeAll()
        mutex.unlock()
        items.forEach { $0(result) }
    }
    
    // MARK: - Transform
    
    open func transform<T>(_ transform: @escaping (OutputResult) throws -> T) -> CoFuture<T> {
        CoTransformFuture(parent: self, transform: transform)
    }
    
}

extension CoFuture {
    
    // MARK: - Cancel
    
    @inlinable public var isCancelled: Bool {
        if case .failure(let error as FutureError) = result {
            return error == .cancelled
        }
        return false
    }
    
    @inlinable public func cancel() {
        send(result: .failure(FutureError.cancelled))
    }
    
    // MARK: - Completion
    
    @usableFromInline func setCompletion(for key: AnyHashable, completion: Completion?) {
        mutex.lock()
        completions[key] = completion
        mutex.unlock()
    }
    
    @inlinable func addCompletion(completion: @escaping Completion) {
        withUnsafePointer(to: completion) {
            setCompletion(for: $0, completion: completion)
        }
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
