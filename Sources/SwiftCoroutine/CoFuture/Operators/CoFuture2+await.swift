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
        mutex?.lock()
        if let result = _result {
            mutex?.unlock()
            return try result.get()
        }
        return try Coroutine.current()
            .await { (callback: @escaping (Result<Value, Error>) -> Void) in
                self.append(callback: callback)
                self.mutex?.unlock()
        }.get()
    }
    
}
