//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 01.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

/// Additional struct with utility methods to work with coroutines.
///
/// - Important: All methods must be called inside a coroutine,
/// otherwise `CoroutineError.mustBeCalledInsideCoroutine` will be thrown.
///
public struct Coroutine {
    
    // MARK: - await
    
    /// Suspends a coroutine поки не буде викликаний callback.
    /// ```
    /// queue.startCoroutine {
    ///     try Coroutine.await { callback in
    ///         someAsyncFunc { callback() }
    ///     }
    /// }
    /// ```
    /// - Parameter callback: The callback для resume coroutine.
    /// - Throws: CoroutineError.mustBeCalledInsideCoroutine якщо метод був викликаний за межами коротини.
    @inlinable public static func await(_ callback: (@escaping () -> Void) -> Void) throws {
        try current().await { completion in callback { completion(()) } }
    }
    
    /// Suspends a coroutine and resumes it on callback.
    /// ```
    /// queue.startCoroutine {
    ///     let result = try Coroutine.await { callback in
    ///         someAsyncFunc { result in callback(result) }
    ///     }
    /// }
    /// ```
    /// - Parameter callback: The callback for resuming a coroutine.
    /// - Throws: `CoroutineError.mustBeCalledInsideCoroutine` if the method is called outside a coroutine.
    /// - Returns: The result which is passed to callback.
    @inlinable public static func await<T>(_ callback: (@escaping (T) -> Void) -> Void) throws -> T {
        try current().await(callback)
    }
    
    /// Suspends a coroutine and resumes it on callback.
    /// ```
    /// queue.startCoroutine {
    ///     let (a, b) = try Coroutine.await { callback in
    ///         someAsyncFunc(callback: callback)
    ///     }
    /// }
    /// ```
    /// - Parameter callback: The callback для resume coroutine.
    /// - Throws: `CoroutineError.mustBeCalledInsideCoroutine` if the method is called outside a coroutine.
    /// - Returns: The result which is passed to callback.
    @inlinable public static func await<T, N>(_ callback: (@escaping (T, N) -> Void) -> Void) throws -> (T, N) {
        try current().await { completion in callback { a, b in completion((a, b)) } }
    }
    
    /// Suspends a coroutine and resumes it on callback.
    /// ```
    /// queue.startCoroutine {
    ///     let (a, b, c) = try Coroutine.await { callback in
    ///         someAsyncFunc(callback: callback)
    ///     }
    /// }
    /// ```
    /// - Parameter callback: The callback для resume coroutine.
    /// - Throws: `CoroutineError.mustBeCalledInsideCoroutine` if the method is called outside a coroutine.
    /// - Returns: The result which is passed to callback.
    @inlinable public static func await<T, N, M>(_ callback: (@escaping (T, N, M) -> Void) -> Void) throws -> (T, N, M) {
        try current().await { completion in callback { a, b, c in completion((a, b, c)) } }
    }
    
    // MARK: - delay
    
    /// Suspends a coroutine for a certain time.
    /// ```
    /// queue.startCoroutine {
    ///     while !someCondition() {
    ///         try Coroutine.delay(.seconds(1))
    ///     }
    /// }
    /// ```
    /// - Parameter time: The time interval for which a coroutine will be suspended.
    /// - Throws: CoroutineError.mustBeCalledInsideCoroutine якщо метод був викликаний за межами коротини.
    @inlinable public static func delay(_ time: DispatchTimeInterval) throws {
        var timer: DispatchSourceTimer!
        try await {
            timer = DispatchSource.createTimer(timeout: .now() + time, handler: $0)
            if #available(OSX 10.12, iOS 10.0, *) {
                timer.activate()
            } else {
                timer.resume()
            }
        }
    }
    
}
