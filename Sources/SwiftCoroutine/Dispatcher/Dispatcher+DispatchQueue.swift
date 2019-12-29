//
//  Dispatcher+DispatchQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension Coroutine.Dispatcher {
    
    @inlinable public static var current: Dispatcher {
        Thread.isMainThread ? .main : .global
    }
    
    @inlinable public static func dispatchQueue(_ queue: DispatchQueue, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], group: DispatchGroup? = nil) -> Dispatcher {
        Dispatcher { queue.async(group: group, qos: qos, flags: flags, execute: $0) }
    }
    
}

extension DispatchQueue {
    
    // MARK: - Async
    
    @inlinable public func async<T>(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], execute work: @escaping () throws -> T) -> CoFuture<T> {
        let dispatcher = Coroutine.Dispatcher
            .dispatchQueue(self, qos: qos, flags: flags, group: group)
        return SwiftCoroutine.async(on: dispatcher, execute: work)
    }
    
    // MARK: - Coroutine
    
    @inlinable public func coroutine(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], stackSize: Coroutine.StackSize = .recommended, execute work: @escaping () throws -> Void) {
        let dispatcher = Coroutine.Dispatcher
            .dispatchQueue(self, qos: qos, flags: flags, group: group)
        SwiftCoroutine.coroutine(on: dispatcher, stackSize: stackSize, execute: work)
    }
    
    @inlinable public func coroutine<T>(group: DispatchGroup? = nil, qos: DispatchQoS = .unspecified, flags: DispatchWorkItemFlags = [], stackSize: Coroutine.StackSize = .recommended, execute work: @escaping () throws -> T) -> CoFuture<T> {
        let dispatcher = Coroutine.Dispatcher
            .dispatchQueue(self, qos: qos, flags: flags, group: group)
        return SwiftCoroutine.coroutine(on: dispatcher, stackSize: stackSize, execute: work)
    }
    
}
