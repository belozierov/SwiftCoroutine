//
//  DispatchQueue+Extensions.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension DispatchQueue {
    
    public func async<T>(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], execute work: @escaping () throws -> T) -> CoFuture<T> {
        let item = CoPromise<T>()
        async(group: group, qos: qos, flags: flags) { item.perform(work) }
        return item
    }
    
}

@inlinable
public func async(on queue: DispatchQueue = .global(),
                  execute work: @escaping () -> Void) {
    queue.async(execute: work)
}

@inlinable
public func async<T>(on queue: DispatchQueue = .global(),
                     execute work: @escaping () throws -> T) -> CoFuture<T> {
    queue.async(execute: work)
}
