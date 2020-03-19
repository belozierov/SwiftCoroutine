//
//  CoFuture+wait.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Dispatch

extension CoFuture {
    
    // MARK: - wait
    /// Wait for the resolution by blocking the current thread until it resolves.
    /// - Parameter timeout: The latest time to wait for the value.
    /// - returns: The value when it completes.
    /// - throws: The error value if it errors.
    public func wait(timeout: DispatchTime? = nil) throws -> Value {
        assert(!Coroutine.isInsideCoroutine, "Use await inside coroutine")
        let group = DispatchGroup()
        group.enter()
        whenComplete { _ in group.leave() }
        if let timeout = timeout {
            if group.wait(timeout: timeout) == .timedOut {
                throw CoFutureError.timeout
            }
        } else {
            group.wait()
        }
        return try _result!.get()
    }
    
}
