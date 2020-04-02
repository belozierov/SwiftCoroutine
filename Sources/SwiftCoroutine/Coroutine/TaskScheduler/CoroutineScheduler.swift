//
//  CoroutineScheduler.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

/// A protocol that defines how to execute a task.
///
/// This protocol has extension methods that allow to launch coroutines on a current scheduler.
/// Inside the coroutine you can use such methods as `Coroutine.await(_:)`, `CoFuture.await()`,
/// and `CoroutineScheduler.await(_:)` to suspend the coroutine without blocking a thread
/// and resume it when the result is ready.
///
/// The framework includes the implementation of this protocol for `DispatchQueue`
/// and you can easily make the same for other schedulers as well.
/// ```
/// extension OperationQueue: CoroutineScheduler {
///
///     public func scheduleTask(_ task: @escaping () -> Void) {
///         addOperation(task)
///     }
///
/// }
/// ```
public protocol CoroutineScheduler {
    
    /// Performs the task at the next possible opportunity.
    func scheduleTask(_ task: @escaping () -> Void)
    
}

extension CoroutineScheduler {
    
    @inlinable internal func startCoroutine(_ task: @escaping () -> Void) {
        CoroutineDispatcher.default.execute(on: self, task: task)
    }
    
    /// Start a new coroutine on the current scheduler.
    ///
    /// As an example, with `Coroutine.await(_:)` you can wrap asynchronous functions with callbacks
    /// to synchronously receive its result without blocking the thread.
    /// ```
    /// //start new coroutine on the main thread
    /// DispatchQueue.main.startCoroutine {
    ///     //execute someAsyncFunc() and await result from its callback
    ///     let result = try Coroutine.await { someAsyncFunc(callback: $0) }
    /// }
    /// ```
    /// - Parameter task: The closure that will be executed inside coroutine.
    /// If the task throws an error, then the coroutine will be terminated.
    @inlinable public func startCoroutine(_ task: @escaping () throws -> Void) {
        startCoroutine { try? task() }
    }
    
    /// Start a coroutine and await its result. Must be called inside other coroutine.
    ///
    /// This method allows to execute given task on other scheduler and await its result without blocking the thread.
    /// ```
    /// //start coroutine on the main thread
    /// DispatchQueue.main.startCoroutine {
    ///     //execute someSyncFunc() on global queue and await its result
    ///     let result = try DispatchQueue.global().await { someSyncFunc() }
    /// }
    /// ```
    /// - Parameter task: The closure that will be executed inside coroutine.
    /// - Throws: Rethrows an error from the task or
    /// throws `CoroutineError.mustBeCalledInsideCoroutine` if the method is called outside a coroutine.
    /// - Returns: Returns the result of the task.
    @inlinable public func await<T>(_ task: () throws -> T) throws -> T {
        try Coroutine.current().await(on: self, task: task)
    }
    
    /// Starts a new coroutine and returns its future result.
    ///
    /// This method allows to execute a given task asynchronously and return `CoFuture` with its future result immediately.
    /// ```
    /// //execute someSyncFunc() on global queue and return its future result
    /// let future = DispatchQueue.global().coroutineFuture { someSyncFunc() }
    /// //start coroutine on main thread
    /// DispatchQueue.main.startCoroutine {
    ///     //await result of future
    ///     let result = try future.await()
    /// }
    /// ```
    /// - Parameter task: The closure that will be executed inside the coroutine.
    /// - Returns: Returns `CoFuture` with the future result of the task.
    @inlinable public func coroutineFuture<T>(_ task: @escaping () throws -> T) -> CoFuture<T> {
        let promise = CoPromise<T>()
        startCoroutine { promise.complete(with: Result(catching: task)) }
        return promise
    }
    
}
