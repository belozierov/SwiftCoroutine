//
//  CoFuture.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

public class CoFuture<Output> {
    
    let mutex: NSRecursiveLock
    @RefBox var resultStorage: OutputResult?
    @ArcRefBox var subscriptions: [AnyHashable: OutputHandler]?
    
    init(mutex: NSRecursiveLock = .init(),
         resultStorage: RefBox<OutputResult?> = .init(),
         subscriptions: ArcRefBox<[AnyHashable: OutputHandler]> = .init(value: [:])) {
        self.mutex = mutex
        _resultStorage = resultStorage
        _subscriptions = subscriptions
    }
    
    @inlinable public func cancel() {
        complete(with: .failure(CoFutureError.cancelled))
    }
    
}

extension CoFuture {
    
    // MARK: - Result
    
    public var result: OutputResult? {
        mutex.lock()
        defer { mutex.unlock() }
        return resultStorage
    }
    
    @inlinable public var isCancelled: Bool {
        if case .failure(let error as CoFutureError) = result {
            return error == .cancelled
        }
        return false
    }
    
    @usableFromInline func complete(with result: OutputResult) {
        mutex.lock()
        guard resultStorage == nil
            else { return mutex.unlock() }
        resultStorage = result
        let handlers = subscriptions
        subscriptions?.removeAll()
        mutex.unlock()
        handlers?.values.forEach { $0(result) }
    }
    
    // MARK: - Identifier
    
    @inlinable public var identifier: Int {
        unsafeBitCast(self, to: Int.self)
    }
    
}

extension CoFuture: CoPublisher {
    
    public typealias Output = Output
    
    public func subscribe(with identifier: AnyHashable, handler: @escaping OutputHandler) {
        mutex.lock()
        if let result = resultStorage {
            mutex.unlock()
            return handler(result)
        }
        subscriptions?[identifier] = handler
        mutex.unlock()
    }
    
    @discardableResult
    public func unsubscribe(_ identifier: AnyHashable) -> OutputHandler? {
        mutex.lock()
        defer { mutex.unlock() }
        return subscriptions?.removeValue(forKey: identifier)
    }
    
}

extension CoFuture: CoCancellable {}

extension CoFuture: Hashable {
    
    @inlinable public static func == (lhs: CoFuture, rhs: CoFuture) -> Bool {
        lhs === rhs
    }
    
    @inlinable public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}


