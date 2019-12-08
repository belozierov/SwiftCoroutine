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
    private var dispatcher: Dispatcher
    private var caller: Coroutine?
    
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
        dispatcher {
            self.setCurrent()
            self.postSuspend(finished: block())
            self.resetCurrent()
        }
    }
    
    // MARK: - Suspend
    
    @inline(__always) open func suspend() {
        resetCurrent()
        context.suspend()
    }
    
    // MARK: - Dispatcher
    
    open func setDispatcher(_ dispatcher: @escaping Dispatcher) {
        self.dispatcher = dispatcher
        if Thread.current.currentCoroutine !== self { return }
        notifyOnceOnSuspend(handler: resume)
        suspend()
    }
    
}

extension Coroutine {
    
    // MARK: - Notifications
    
    public static let coroutineDidSuspend = Notification.Name(#function)
    public static let coroutineDidComplete = Notification.Name(#function)
    
    private func postSuspend(finished: Bool) {
        let name = finished ? Coroutine.coroutineDidComplete : Coroutine.coroutineDidSuspend
        notificationCenter.post(name: name, object: self)
    }
    
    public func notifyOnceOnSuspend(handler: @escaping Block) {
        notifyOnce(name: Coroutine.coroutineDidSuspend, handler: handler)
    }
    
    public func notifyOnCompletion(handler: @escaping Block) {
        notifyOnce(name: Coroutine.coroutineDidComplete, handler: handler)
    }
    
    private func notifyOnce(name: Notification.Name, handler: @escaping Block) {
        notificationCenter.notifyOnce(name: name, object: self) { _ in handler() }
    }
    
    private var notificationCenter: NotificationCenter { .default }
    
}

extension Coroutine {
    
    // MARK: - Current coroutine lifecycle
    
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

extension Thread {
    
    public var currentCoroutine: Coroutine? {
        @inline(__always) get { threadDictionary.value(forKey: #function) as? Coroutine }
        @inline(__always) set { threadDictionary.setValue(newValue, forKey: #function) }
    }
    
}
