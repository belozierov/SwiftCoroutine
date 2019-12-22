//
//  DispatchQueue+Coroutine.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 01.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension DispatchQueue {
    
    @inlinable public static var _current: DispatchQueue {
        Thread.isMainThread ? .main : .global()
    }

    public func coroutine(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], execute work: @escaping () throws -> Void) {
        Coroutine
            .fromPool { self.async(group: group, qos: qos, flags: flags, execute: $0.block) }
            .start { try? work() }
    }
    
    public func coroutine<T>(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], execute work: @escaping () throws -> T) -> CoFuture<T> {
        let item = CoPromise<T>()
        coroutine(group: group, qos: qos, flags: flags) { item.perform(work) }
        return item
    }
    
    public func setDispatcher(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = []) {
        assert(Coroutine.current != nil, "setDispatcher must be called inside coroutine")
        Coroutine.current?.restart {
            self.async(group: group, qos: qos, flags: flags, execute: $0.block)
        }
    }

}

@inlinable
public func coroutine(on queue: DispatchQueue = ._current,
                      execute work: @escaping () throws -> Void) {
    queue.coroutine(execute: work)
}

@inlinable
public func coroutine<T>(on queue: DispatchQueue = ._current,
                         execute work: @escaping () throws -> T) -> CoFuture<T> {
    queue.coroutine(execute: work)
}
