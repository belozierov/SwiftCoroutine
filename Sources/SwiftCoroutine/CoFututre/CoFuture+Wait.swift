//
//  CoFuture+Wait.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension CoFuture {
    
    public func wait(timeout: DispatchTime? = nil) throws -> Output {
        assert(!Coroutine.isInsideCoroutine, "Use await inside coroutine")
        mutex.lock()
        if let result = result {
            mutex.unlock()
            return try result.get()
        }
        var result: OutputResult!
        let group = DispatchGroup()
        group.enter()
        addCompletion {
            result = $0
            group.leave()
        }
        mutex.unlock()
        if let timeout = timeout {
            if group.wait(timeout: timeout) == .timedOut {
                throw FutureError.timeout
            }
        } else {
            group.wait()
        }
        return try result.get()
    }
    
}
