//
//  CoFuture3+await.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - await
    
    /// Await for the result of this `CoFuture` without blocking the current thread. Must be called inside a coroutine.
    /// ```
    /// //execute someSyncFunc() on global queue and return its future result
    /// let future = DispatchQueue.global().coroutineFuture { someSyncFunc() }
    /// //start coroutine on main thread
    /// DispatchQueue.main.startCoroutine {
    ///     //await result of future
    ///     let result = try future.await()
    /// }
    /// ```
    /// - Throws: The failed result of the `CoFuture`.
    /// - Returns: The value of the `CoFuture` when it is completed.
    public func await() throws -> Value {
        mutex?.lock()
        if let result = _result {
            mutex?.unlock()
            return try result.get()
        }
        return try Coroutine.current.await { (callback: @escaping (Result<Value, Error>) -> Void) in
            self.append(callback: callback)
            self.mutex?.unlock()
        }.get()
    }
    
}
