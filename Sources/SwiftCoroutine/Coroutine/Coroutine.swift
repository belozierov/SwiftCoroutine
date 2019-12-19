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
    public typealias Handler = (Bool) -> Void
    
    @inline(__always) public static var current: Coroutine? {
        Thread.current.currentCoroutine
    }
    
    let context: CoroutineContext
    private var dispatcher: Dispatcher, handler: Handler?
    
    @inline(__always) init(context: CoroutineContext, dispatcher: @escaping Dispatcher) {
        self.context = context
        self.dispatcher = dispatcher
    }
    
    public init(stackSizeInPages: Int = 32, dispatcher: @escaping Dispatcher) {
        self.context = CoroutineContext(stackSizeInPages: stackSizeInPages)
        self.dispatcher = dispatcher
    }
    
    open func addHandler(_ handler: @escaping Handler) {
        self.handler = self.handler.map {
            previous in { previous($0); handler($0) }
        } ?? handler
    }
    
    // MARK: - Perform
    
    @inline(__always) open func start(block: @escaping Block) {
        assert(self !== Coroutine.current, "Start must be called outside current coroutine")
        perform { self.context.start(block: block) }
    }
    
    @inline(__always) open func resume() {
        perform(context.resume)
    }
    
    @inline(__always) open func restart(with dispatcher: @escaping Dispatcher) {
        self.dispatcher = dispatcher
        suspend(with: resume)
    }
    
    private func perform(_ block: @escaping () -> Bool) {
        var caller: Coroutine?
        dispatcher { [unowned self] in
            let thread = Thread.current
            caller = thread.currentCoroutine
            thread.currentCoroutine = self
            let finished = block()
            self.handler?(finished)
            thread.currentCoroutine = caller
        }
    }
    
    // MARK: - Suspend
    
    @inline(__always) open func suspend() {
        assert(self === Coroutine.current, "Suspend must be called inside current coroutine")
        context.suspend()
    }
    
    @inline(__always) open func suspend(with completion: @escaping Block) {
        let previousHandler = handler
        handler = { [unowned self] in
            self.handler = previousHandler
            previousHandler?($0)
            completion()
        }
        suspend()
    }
    
}

extension Thread {
    
    fileprivate var currentCoroutine: Coroutine? {
        @inline(__always) get { threadDictionary.value(forKey: #function) as? Coroutine }
        @inline(__always) set { threadDictionary.setValue(newValue, forKey: #function) }
    }
    
}
