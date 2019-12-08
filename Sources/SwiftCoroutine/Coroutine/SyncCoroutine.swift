//
//  SyncCoroutine.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class SyncCoroutine {
    
    public static func fromPool() -> SyncCoroutine {
        let context = CoroutineContext.pool.pop()
        let coroutine = SyncCoroutine(context: context)
        coroutine.notifyOnCompletion { CoroutineContext.pool.push(context) }
        return coroutine
    }
    
    private let context: CoroutineContext
    private var caller: Coroutine?
    
    @inline(__always)
    public init(context: CoroutineContext) {
        self.context = context
    }
    
    @discardableResult @inline(__always)
    open func start(block: @escaping Block) -> Bool {
        perform { context.start(block: block) }
    }
    
    @inline(__always)
    open func resumeWithHandler() -> Bool {
        perform(context.resume)
    }
    
    @inline(__always)
    open func suspend() {
        resetCurrent()
        context.suspend()
    }
    
    private func perform(_ block: () -> Bool) -> Bool {
        setCurrent()
        let finished = block()
        resetCurrent()
        postSuspend(finished: finished)
        return finished
    }
    
    private func setCurrent() {
        let thread = Thread.current
        caller = thread.currentCoroutine
        thread.currentCoroutine = self
    }
    
    private func resetCurrent() {
        Thread.current.currentCoroutine = caller
        caller = nil
    }
    
}

extension SyncCoroutine: Coroutine {
    
    @inline(__always)
    open func resume() {
        _ = resumeWithHandler()
    }
    
    @inline(__always)
    open func setDispatcher(_ dispatcher: AsyncCoroutine.Dispatcher) {
        guard isCurrent else { fatalError() }
        let coroutine = AsyncCoroutine(context: context, dispatcher: dispatcher)
        notifyOnceOnSuspend(handler: coroutine.resume)
        suspend()
    }
    
}
