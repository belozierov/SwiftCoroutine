//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

open class Coroutine {
    
    public typealias Block = Dispatcher.Block
    public typealias Handler = (Bool) -> Void
    
    public enum CoroutineError: Error {
        case mustBeCalledInsideCoroutine
    }
    
    let context: CoroutineContext
    private var dispatcher: Dispatcher?, handler: Handler?
    
    init(context: CoroutineContext, dispatcher: Dispatcher?) {
        self.context = context
        self.dispatcher = dispatcher
    }
    
    public init(stackSize: StackSize = .recommended, dispatcher: Dispatcher? = nil) {
        assert(stackSize.size >= StackSize.minimal.size,
               "Stack size must be more or equal to minimal")
        self.context = CoroutineContext(stackSize: stackSize.size)
        self.dispatcher = dispatcher
    }
    
    open func addHandler(_ handler: @escaping Handler) {
        self.handler = self.handler.map {
            previous in { previous($0); handler($0) }
        } ?? handler
    }
    
    // MARK: - Perform
    
    @inline(__always) open func start(block: @escaping Block) {
        assert(!isCurrent, "Start must be called outside current coroutine")
        performWithDispatcher { self.context.start(block: block) }
    }
    
    @inline(__always) open func resume() {
        performWithDispatcher(block: context.resume)
    }
    
    @inline(__always) open func restart(with dispatcher: Dispatcher) {
        self.dispatcher = dispatcher
        suspend(with: resume)
    }
    
    private func performWithDispatcher(block: @escaping () -> Bool) {
        (dispatcher?.perform ?? { $0() }) { [unowned self] in
            self.performAsCurrent {
                let finished = block()
                self.handler?(finished)
            }
        }
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
