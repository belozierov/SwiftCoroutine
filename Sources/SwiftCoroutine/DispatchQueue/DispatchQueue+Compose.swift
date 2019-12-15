//
//  FututeComposite.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 23.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

@_functionBuilder
public struct FututeComposite<T> {
    
    public static func buildBlock(_ components: CoFuture<T>...) -> [CoFuture<T>] {
        components
    }
    
}

extension DispatchQueue {
    
    public func compose<T>(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], @FututeComposite<T> builder: @escaping () -> [CoFuture<T>]) -> CoFuture<[T]> {
        let promise = CoPromise<[T]>()
        coroutine(group: group, qos: qos, flags: flags) {
            promise.perform { try builder().map { try $0.await() } }
        }
        return promise
    }
    
}

@inlinable public
func compose<T>(on queue: DispatchQueue = .__current, @FututeComposite<T> builder: @escaping () -> [CoFuture<T>]) -> CoFuture<[T]> {
    queue.compose(builder: builder)
}
