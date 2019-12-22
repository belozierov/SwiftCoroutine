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
    private var _completions = [AnyHashable: Completion]()
    
    // MARK: - Result
    
    open var result: OutputResult? { nil }
    
    @usableFromInline func send(result: OutputResult) {
        mutex.lock()
        let items = _completions.values
        _completions.removeAll()
        mutex.unlock()
        items.forEach { $0(result) }
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
    
    // MARK: - Completions
    
    @usableFromInline var completions: [AnyHashable: Completion] {
        get {
            mutex.lock()
            defer { mutex.unlock() }
            return _completions
        }
        set {
            mutex.lock()
            _completions = newValue
            mutex.unlock()
        }
    }
    
    @inlinable func addCompletion(completion: @escaping Completion) {
        withUnsafePointer(to: completion) { completions[$0] = completion }
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
