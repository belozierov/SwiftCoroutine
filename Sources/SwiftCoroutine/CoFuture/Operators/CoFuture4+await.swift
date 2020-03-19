//
//  CoFuture+await.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    // MARK: - await
    
    public func await() throws -> Value {
        let coroutine = try Coroutine.current()
        lock()
        if let result = _result {
            unlock()
            return try result.get()
        }
        append { _ in try? coroutine.resume() }
        try coroutine.coroutine.suspend(with: unlock)
        if let result = _result { return try result.get() }
        throw CoroutineError.wrongState
    }
    
}
