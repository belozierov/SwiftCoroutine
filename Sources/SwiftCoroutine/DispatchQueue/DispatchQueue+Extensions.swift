//
//  DispatchQueue+Extensions.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension DispatchQueue {
    
    public static var __current: DispatchQueue {
        Thread.isMainThread ? .main : .global()
    }
    
    open func async<T>(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], execute work: @escaping () throws -> T) -> CoFuture<T> {
        let item = CoPromise<T>()
        async(group: group, qos: qos, flags: flags) { item.perform(work) }
        return item
    }
    
}

@inline(__always)
public func async(on queue: DispatchQueue = .__current, execute work: @escaping () -> Void) {
    queue.async(execute: work)
}

@inline(__always)
public func async<T>(on queue: DispatchQueue = .__current, execute work: @escaping () throws -> T) -> CoFuture<T> {
    queue.async(execute: work)
}
