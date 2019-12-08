//
//  AsyncCoroutine.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class AsyncCoroutine: Coroutine {
    
    public struct Dispatcher {
        let queue: DispatchQueue, group: DispatchGroup?, qos: DispatchQoS, flags: DispatchWorkItemFlags
    }
    
    public static func fromPool(with dispatcher: Dispatcher) -> AsyncCoroutine {
        let context = CoroutineContext.pool.pop()
        let coroutine = AsyncCoroutine(context: context, dispatcher: dispatcher)
        coroutine.notifyOnCompletion { CoroutineContext.pool.push(context) }
        return coroutine
    }
    
    private let context: CoroutineContext
    private(set) var dispatcher: Dispatcher
    
    public init(context: CoroutineContext, dispatcher: Dispatcher) {
        self.context = context
        self.dispatcher = dispatcher
    }
    
    @inline(__always)
    open func start(block: @escaping Block) {
        perform { self.context.start(block: block) }
    }
    
    @inline(__always)
    open func resume() {
        perform(context.resume)
    }
    
    @inline(__always)
    open func suspend() {
        Thread.current.currentCoroutine = nil
        context.suspend()
    }
    
    open func setDispatcher(_ dispatcher: AsyncCoroutine.Dispatcher) {
        self.dispatcher = dispatcher
        guard isCurrent else { return }
        notifyOnSuspend(handler: resume)
        suspend()
    }
    
    private func perform(_ work: @escaping () -> Bool) {
        dispatcher.perform {
            Thread.current.currentCoroutine = self
            self.postSuspend(finished: work())
            Thread.current.currentCoroutine = nil
        }
    }
    
}

extension AsyncCoroutine.Dispatcher {
    
    @inline(__always)
    public func perform(_ block: @escaping AsyncCoroutine.Block) {
        queue.async(group: group, qos: qos, flags: flags, execute: block)
    }
    
}
