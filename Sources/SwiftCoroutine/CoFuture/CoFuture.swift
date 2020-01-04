//
//  CoFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

public class CoFuture<Output> {
    
    public enum FutureError: Error {
        case cancelled, timeout
    }
    
    let mutex: NSRecursiveLock
    private var subscriptions = [AnyHashable: OutputHandler]()
    
    @usableFromInline init(mutex: NSRecursiveLock = .init()) {
        self.mutex = mutex
    }
    
    @inlinable public var result: OutputResult? { nil }
    @inlinable func saveResult(_ result: OutputResult) {}
    
}

extension CoFuture {
    
    @usableFromInline func complete(with result: OutputResult) {
        mutex.lock()
        saveResult(result)
        let items = subscriptions
        subscriptions.removeAll()
        mutex.unlock()
        items.values.forEach { $0(result) }
    }
    
}

extension CoFuture: CoCancellable {
    
    @inlinable public func cancel() {
        complete(with: .failure(FutureError.cancelled))
    }
    
    @inlinable public var isCancelled: Bool {
        if case .failure(let error as FutureError) = result {
            return error == .cancelled
        }
        return false
    }
    
}

extension CoFuture: CoPublisher {
    
    public typealias Output = Output
    
    public func subscribe(with identifier: AnyHashable, handler: @escaping OutputHandler) {
        subscriptions[identifier] = handler
    }
    
    public func unsubscribe(_ identifier: AnyHashable) {
        subscriptions[identifier] = nil
    }
    
}

extension CoFuture: Hashable {
    
    @inlinable public static func == (lhs: CoFuture, rhs: CoFuture) -> Bool {
        lhs === rhs
    }
    
    @inlinable public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
