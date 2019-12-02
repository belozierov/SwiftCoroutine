//
//  Publisher+Extensions.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Combine

@available(OSX 10.15, iOS 13.0, *)
extension Publisher {
    
    public func await() throws -> Output {
        let promise = CoPromise<Output>()
        var subscriptions = Set<AnyCancellable>()
        mapError { $0 }.subscribe(promise).store(in: &subscriptions)
        return try promise.await()
     }
    
}
