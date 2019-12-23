//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

open class Coroutine {
    
    public typealias Block = Dispatcher.Block
    public typealias Handler = (Bool) -> Void
    
    public enum CoroutineError: Error {
        case mustBeCalledInsideCoroutine
    }
    
    @inline(__always) public static func current() throws -> Coroutine {
        if let coroutine = Thread.current.currentCoroutine { return coroutine }
        throw CoroutineError.mustBeCalledInsideCoroutine
    }
    
    @inlinable public static var isInsideCoroutine: Bool {
        (try? current()) != nil
    }
    
    let context: CoroutineContext
    private var dispatcher: Dispatcher?, handler: Handler?
    
    init(context: CoroutineContext, dispatcher: Dispatcher?) {
        self.context = context
        self.dispatcher = dispatcher
    }
    
    public init(dispatcher: Dispatcher? = nil) {
        self.context = CoroutineContext()
        self.dispatcher = dispatcher
    }
    
    public init(stackSizeInPages pages: Int, dispatcher: Dispatcher) {
        self.context = CoroutineContext(stackSizeInPages: pages)
        self.dispatcher = dispatcher
    }
    
    open func addHandler(_ handler: @escaping Handler) {
        self.handler = self.handler.map {
            previous in { previous($0); handler($0) }
        } ?? handler
    }
    
    @inlinable open var isCurrent: Bool {
        self === (try? Coroutine.current())
    }
    
    // MARK: - Perform
    
    @inline(__always) open func start(block: @escaping Block) {
        assert(!isCurrent, "Start must be called outside current coroutine")
        perform(fromSuspend: false) { self.context.start(block: block) }
    }
    
    @inline(__always) open func resume() {
        perform(fromSuspend: true, block: context.resume)
    }
    
    @inline(__always) open func restart(with dispatcher: Dispatcher) {
        self.dispatcher = dispatcher
        suspend(with: resume)
    }
    
    private func perform(fromSuspend: Bool, block: @escaping () -> Bool) {
        var caller: Coroutine?
        let performer = { [unowned self] in
            let thread = Thread.current
            caller = thread.currentCoroutine
            thread.currentCoroutine = self
            let finished = block()
            self.handler?(finished)
            thread.currentCoroutine = caller
        }
        dispatcher?.perform(work: performer) ?? performer()
    }
    
    // MARK: - Suspend
    
    @inline(__always) open func suspend() {
        assert(isCurrent, "Suspend must be called inside current coroutine")
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

extension Coroutine: Hashable {
    
    public static func == (lhs: Coroutine, rhs: Coroutine) -> Bool {
        lhs === rhs
    }
    
    public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}

extension Thread {
    
    fileprivate var currentCoroutine: Coroutine? {
        @inline(__always) get { threadDictionary.value(forKey: #function) as? Coroutine }
        @inline(__always) set { threadDictionary.setValue(newValue, forKey: #function) }
    }
    
}
