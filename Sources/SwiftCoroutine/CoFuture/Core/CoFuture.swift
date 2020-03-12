//
//  CoFuture2.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Darwin

protocol _CoFutureCancellable: class {
    
    func cancel()
    
}

public class CoFuture<Value> {
    
    private let mutex: Mutex?
    private var parent: UnownedCancellable?
    private var callbacks: ContiguousArray<Child>?
    final private(set) var _result: Optional<Result<Value, Error>>
    
    @usableFromInline init(mutex: Mutex?, result: Result<Value, Error>?) {
        self.mutex = mutex
        _result = result
    }
    
    deinit {
        callbacks?.forEach { $0.callback(.failure(CoFutureError.cancelled)) }
        destroyMutex()
    }
    
}

extension CoFuture {
    
    @inlinable public convenience init(result: Result<Value, Error>) {
        self.init(mutex: nil, result: result)
    }
    
    @inlinable public convenience init(value: Value) {
        self.init(result: .success(value))
    }
    
    @inlinable public convenience init(error: Error) {
        self.init(result: .failure(error))
    }
    
    // MARK: - Mutex
    
    @usableFromInline typealias Mutex = UnsafeMutablePointer<pthread_mutex_t>
    
    func lock() {
        if let mutex = mutex { pthread_mutex_lock(mutex) }
    }
    
    func unlock() {
        if let mutex = mutex { pthread_mutex_unlock(mutex) }
    }
    
    private func destroyMutex() {
        guard let mutex = mutex else { return }
        pthread_mutex_destroy(mutex)
        mutex.deallocate()
    }
    
    // MARK: - Result
    
    public var result: Result<Value, Error>? {
        lock()
        defer { unlock() }
        return _result
    }
    
    @usableFromInline func setResult(_ result: Result<Value, Error>) {
        lock()
        if _result != nil { return unlock() }
        lockedComplete(with: result)
    }
    
    private func lockedComplete(with result: Result<Value, Error>) {
        _result = result
        unlock()
        callbacks?.forEach { $0.callback(result) }
        callbacks = nil
        parent = nil
    }
    
    // MARK: - Callback
    
    typealias Callback = (Result<Value, Error>) -> Void
    private struct Child { let callback: Callback }
    
    func append(callback: @escaping Callback) {
        callbacks.append(.init(callback: callback))
    }
    
    func addChild<T>(future: CoFuture<T>, callback: @escaping Callback) {
        future.parent = .init(cancellable: self)
        append(callback: callback)
    }
    
}

extension CoFuture: _CoFutureCancellable {
    
    private struct UnownedCancellable {
        unowned(unsafe) let cancellable: _CoFutureCancellable
    }
    
    // MARK: - cancel
    
    public func cancel() {
        if let parent = parent {
            parent.cancellable.cancel()
        } else {
            setResult(.failure(CoFutureError.cancelled))
        }
    }
    
    @inlinable public var isCanceled: Bool {
        if case .failure(let error as CoFutureError)? = result {
            return error == .cancelled
        }
        return false
    }
    
}

extension CoPromise {
    
    @inlinable public convenience init() {
        let mutex = Mutex.allocate(capacity: 1)
        pthread_mutex_init(mutex, nil)
        self.init(mutex: mutex, result: nil)
    }
    
}

extension CoFuture: Hashable {
    
    // MARK: - Hashable
    
    @inlinable public static func == (lhs: CoFuture, rhs: CoFuture) -> Bool {
        lhs === rhs
    }
    
    @inlinable public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}

extension Optional {
    
    fileprivate mutating func append<T>(_ element: T) where Wrapped == ContiguousArray<T> {
        if self == nil {
            self = [element]
        } else {
            self!.append(element)
        }
    }
    
}
