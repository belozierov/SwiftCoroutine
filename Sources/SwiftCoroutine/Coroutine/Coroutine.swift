//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class Coroutine {
    
    public typealias Block = () -> Void
    public typealias Dispatcher = (@escaping Block) -> Void
    
    public static func fromPool(with dispatcher: @escaping Dispatcher) -> Coroutine {
        let context = CoroutineContext.pool.pop()
        let coroutine = Coroutine(context: context, dispatcher: dispatcher)
        coroutine.notifyOnCompletion { CoroutineContext.pool.push(context) }
        return coroutine
    }
    
    private let context: CoroutineContext
    private var suspendHandlers = [Block]()
    private var completionHandlers = [Block]()
    var dispatcher: Dispatcher
    
    public init(context: CoroutineContext, dispatcher: @escaping Dispatcher) {
        self.context = context
        self.dispatcher = dispatcher
    }
    
    // MARK: - Perform
    
    @inline(__always) open func start(block: @escaping Block) {
        perform { self.context.start(block: block) }
    }
    
    @inline(__always) open func resume() {
        perform(context.resume)
    }
    
    private func perform(_ block: @escaping () -> Bool) {
        var caller: Coroutine?
        dispatcher { [unowned self] in
            let thread = Thread.current
            caller = thread.currentCoroutine
            thread.currentCoroutine = self
            self.postSuspend(finished: block())
            thread.currentCoroutine = caller
        }
    }
    
    // MARK: - Suspend
    
    @inline(__always) open func suspend() {
        context.suspend()
    }
    
}

extension Coroutine {
    
    // MARK: - Notifications
    
    @inline(__always) public func notifyOnceOnSuspend(handler: @escaping Block) {
        suspendHandlers.append(handler)
    }
    
    @inline(__always) public func notifyOnCompletion(handler: @escaping Block) {
        completionHandlers.append(handler)
    }
    
    private func postSuspend(finished: Bool) {
        let handlers: [Block]
        if finished {
            handlers = completionHandlers
            completionHandlers.removeAll()
        } else {
            handlers = suspendHandlers
        }
        suspendHandlers.removeAll()
        handlers.forEach { $0() }
    }
    
}

extension Thread {
    
    public var currentCoroutine: Coroutine? {
        @inline(__always) get { threadDictionary.value(forKey: #function) as? Coroutine }
        @inline(__always) set { threadDictionary.setValue(newValue, forKey: #function) }
    }
    
}
