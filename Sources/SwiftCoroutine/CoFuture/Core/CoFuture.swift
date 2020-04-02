//
//  CoFuture2.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

/// Holder for a result that will be provided later.
///
/// `CoFuture` and its subclass `CoPromise` are the implementation of the Future/Promise approach.
/// They allow to launch asynchronous tasks and immediately return` CoFuture` with its future results.
/// The available result can be observed by the `whenComplete()` callback
/// or by `await()` inside a coroutine without blocking a thread.
///
/// ```
/// func makeFutureOne(args) -> CoFuture<Response> {
///     let promise = CoPromise<Response>()
///     someAsyncFuncWithCallback { response in
///         . . . do some work . . .
///         promise.success(response)
///     }
///     return promise
/// }
///
/// func makeFutureTwo(args) -> CoFuture<Response> {
///     queue.coroutineFuture {
///         let future = makeFutureOne(args)
///         . . . do some work . . .
///         let response = try future.await()
///         . . . create result using response . . .
///         return result
///     }
///  }
///
/// func performSomeWork(args) {
///     let future = makeFutureTwo(args)
///     mainQueue.startCoroutine {
///         . . . do some work . . .
///         let result = try future.await()
///         . . . do some work using result . . .
///     }
/// }
/// ```
public class CoFuture<Value> {
    
    internal let mutex: PsxLock?
    private var callbacks: ContiguousArray<Child>?
    final private(set) var _result: Optional<Result<Value, Error>>
    
    @usableFromInline internal init(mutex: PsxLock?, result: Result<Value, Error>?) {
        self.mutex = mutex
        _result = result
    }
    
    deinit {
        callbacks?.forEach { $0.callback(.failure(CoFutureError.canceled)) }
        mutex?.free()
    }
    
}

extension CoFuture {
    
    /// Initializes a future with result.
    /// - Parameter result: The result provided by this future.
    @inlinable public convenience init(result: Result<Value, Error>) {
        self.init(mutex: nil, result: result)
    }
    
    /// Initializes a future with success value.
    /// - Parameter value: The value provided by this future.
    @inlinable public convenience init(value: Value) {
        self.init(result: .success(value))
    }
    
    /// Initializes a future with error.
    /// - Parameter error: The error provided by this future.
    @inlinable public convenience init(error: Error) {
        self.init(result: .failure(error))
    }
    
    // MARK: - result
    
    /// Returns completed result or nil if this future has not been completed yet.
    public var result: Result<Value, Error>? {
        mutex?.lock()
        defer { mutex?.unlock() }
        return _result
    }
    
    @usableFromInline internal func setResult(_ result: Result<Value, Error>) {
        mutex?.lock()
        if _result != nil {
            mutex?.unlock()
        } else {
            _result = result
            mutex?.unlock()
            callbacks?.forEach { $0.callback(result) }
            callbacks = nil
        }
    }
    
    // MARK: - Callback
    
    internal typealias Callback = (Result<Value, Error>) -> Void
    private struct Child { let callback: Callback }
    
    internal func append(callback: @escaping Callback) {
        if callbacks == nil {
            callbacks = [.init(callback: callback)]
        } else {
            callbacks?.append(.init(callback: callback))
        }
    }
    
    // MARK: - cancel

    /// Returns `true` when the current future is canceled.
    @inlinable public var isCanceled: Bool {
        if case .failure(let error as CoFutureError)? = result {
            return error == .canceled
        }
        return false
    }
    
    /// Cancels the current future.
    @inlinable public func cancel() {
        setResult(.failure(CoFutureError.canceled))
    }
    
}
