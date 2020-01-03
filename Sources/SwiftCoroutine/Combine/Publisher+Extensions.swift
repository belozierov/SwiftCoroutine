//
//  Publisher+Extensions.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Combine
import Dispatch

@available(OSX 10.15, iOS 13.0, *)
extension Publisher {
    
    @inline(__always) public func await() throws -> Output {
        try coFuture.future.await()
     }
    
    @inline(__always) public func await(timeout: DispatchTime) throws -> Output {
        try coFuture.future.await(timeout: timeout)
    }
    
    private var coFuture: (future: CoFuture<Output>, cancellable: AnyCancellable)  {
        let promise = CoPromise<Output>()
        return (promise, mapError { $0 }.subscribe(promise))
    }
    
}
