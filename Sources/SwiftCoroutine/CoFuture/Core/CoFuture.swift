//
//  CoFuture2.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

private protocol _CoFutureCancellable: class {

    func cancel()
    
}

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
///
/// For coroutine error handling you can use standart `do-catch` statement or use `CoFuture` as an alternative.
///
/// ```
/// //execute coroutine and return CoFuture<Void> that we will use for error handling
/// DispatchQueue.main.coroutineFuture {
///     let result = try makeSomeFuture().await()
///     . . . use result . . .
/// }.whenFailure { error in
///     . . . handle error . . .
/// }
/// ```
///
/// Apple has introduced a new reactive programming framework `Combine`
/// that makes writing asynchronous code easier and includes a lot of convenient and common functionality.
/// We can use it with coroutines by making `CoFuture` a subscriber and await its result.
///
/// ```
/// //create Combine publisher
/// let publisher = URLSession.shared.dataTaskPublisher(for: url).map(\.data)
///
/// //execute coroutine on the main thread
/// DispatchQueue.main.startCoroutine {
///     //subscribe CoFuture to publisher
///     let future = publisher.subscribeCoFuture()
///
///     //await data without blocking the thread
///     let data: Data = try future.await()
/// }
/// ```
///
public class CoFuture<Value> {
    
    private var resultState: Int
    private var nodes: CallbackStack<Result<Value, Error>>
    private var _result: Optional<Result<Value, Error>>
    private var parent: UnownedCancellable?
    
    @usableFromInline internal init(_result: Result<Value, Error>?) {
        if let result = _result {
            self._result = result
            resultState = 1
            nodes = CallbackStack(isFinished: true)
        } else {
            self._result = nil
            resultState = 0
            nodes = CallbackStack()
        }
    }
    
    deinit {
        if !nodes.isEmpty {
            nodes.finish(with: .failure(CoFutureError.canceled))
        }
    }
    
}

extension CoFuture: _CoFutureCancellable {

    /// Starts a new coroutine and initializes future with its result.
    ///
    /// - Note: If you cancel this `CoFuture`, it will also cancel the coroutine that was started inside of it.
    ///
    /// ```
    /// func sum(future1: CoFuture<Int>, future2: CoFuture<Int>) -> CoFuture<Int> {
    ///     CoFuture { try future1.await() + future2.await() }
    /// }
    /// ```
    /// - Parameter task: The closure that will be executed inside the coroutine.
    @inlinable public convenience init(task: @escaping () throws -> Value) {
        self.init(_result: nil)
        Coroutine.start {
            let current = try? Coroutine.current()
            self.whenCanceled { current?.cancel() }
            self.setResult(Result(catching: task))
        }
    }
    
    /// Initializes a future with result.
    /// - Parameter result: The result provided by this future.
    @inlinable public convenience init(result: Result<Value, Error>) {
        self.init(_result: result)
    }
    
    // MARK: - result
    
    /// Returns completed result or nil if this future has not been completed yet.
    public var result: Result<Value, Error>? {
        nodes.isClosed ? _result : nil
    }
    
    @usableFromInline internal func setResult(_ result: Result<Value, Error>) {
        guard atomicExchange(&resultState, with: 1) == 0 else { return }
        _result = result
        parent = nil
        nodes.close()?.finish(with: result)
    }
    
    // MARK: - Callback
    
    @usableFromInline internal typealias Callback = (Result<Value, Error>) -> Void
    
    @usableFromInline internal func addCallback(_ callback: @escaping Callback) {
        if !nodes.append(callback) { _result.map(callback) }
    }
    
    internal func addChild<T>(future: CoFuture<T>, callback: @escaping Callback) {
        future.parent = .init(cancellable: self)
        addCallback(callback)
    }
    
    // MARK: - cancel
    
    private struct UnownedCancellable {
        unowned(unsafe) let cancellable: _CoFutureCancellable
    }

    /// Returns `true` when the current future is canceled.
    @inlinable public var isCanceled: Bool {
        if case .failure(let error as CoFutureError)? = result {
            return error == .canceled
        }
        return false
    }
    
    /// Cancels the current future.
    public func cancel() {
        if let parent = parent {
            parent.cancellable.cancel()
        } else {
            setResult(.failure(CoFutureError.canceled))
        }
    }
    
}
