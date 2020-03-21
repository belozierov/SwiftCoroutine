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
        lock()
        if let result = _result {
            unlock()
            return try result.get()
        }
        return try Coroutine.current().coroutine
            .await { (callback: @escaping (Result<Value, Error>) -> Void) in
                self.append(callback: callback)
                self.unlock()
        }.get()
    }
    
}
