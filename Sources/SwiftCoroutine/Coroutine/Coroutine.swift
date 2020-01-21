//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

final public class Coroutine {
    
    public typealias Block = () -> Void
    public typealias Handler = (Bool) -> Void
    
    public enum CoroutineError: Error {
        case mustBeCalledInsideCoroutine
    }
    
    public enum State: Int {
        case prepared, running, suspended
    }
    
    let context: CoroutineContext
    var subRoutines = [CoSubroutine]()
    private var dispatchBlock: Dispatcher.DispatchBlock
    private var handler: Handler?
    
    @AtomicIntRepresentable
    public private(set) var state: State = .prepared
    
    init(context: CoroutineContext, dispatcher: Dispatcher) {
        self.context = context
        dispatchBlock = dispatcher.dispatchBlock
    }
    
    public convenience
    init(dispatcher: Dispatcher = .sync, stackSize: StackSize = .recommended) {
        assert(stackSize.size >= StackSize.minimal.size,
               "Stack size must be more or equal to minimal")
        let context = CoroutineContext(stackSize: stackSize.size)
        self.init(context: context, dispatcher: dispatcher)
    }
    
    public func addHandler(_ handler: @escaping Handler) {
        self.handler = self.handler.map {
            previous in { previous($0); handler($0) }
        } ?? handler
    }
    
    // MARK: - Start/resume
    
    @inline(__always) public func start(block: @escaping Block) {
        assert(!isCurrent, "Start must be called outside current coroutine")
        assert(state == .prepared, "Start must be called for prepared coroutine")
        perform { self.context.start(block: block) }
    }
    
    @inline(__always) public func resume() {
        assert(state == .suspended, "Resume must be called for suspended coroutine")
        perform(block: context.resume)
    }
    
    // MARK: - Suspend
    
    @inline(__always) public func suspend() {
        assert(isCurrent, "Suspend must be called inside current coroutine")
        assert(state == .running, "Suspend must be called for running coroutine")
        context.suspend()
    }
    
    @inline(__always) public func suspend(with completion: @escaping Block) {
        let previousHandler = handler
        handler = { [unowned self] in
            self.handler = previousHandler
            previousHandler?($0)
            completion()
        }
        suspend()
    }
    
    // MARK: - Dispatcher
    
    @inline(__always) public func restart(with dispatcher: Dispatcher) {
        assert(isCurrent, "Restart must be called inside current coroutine")
        assert(state == .running, "Restart must be called for running coroutine")
        dispatchBlock = dispatcher.dispatchBlock
        suspend(with: resume)
    }
    
    private func perform(block: @escaping () -> Bool) {
        state = .running
        dispatchBlock { [unowned self] in
            self.performAsCurrent {
                let finished = block()
                self.state = finished ? .prepared : .suspended
                self.handler?(finished)
            }
        }
    }
    
}

extension Coroutine: Hashable {
    
    @inlinable public static func == (lhs: Coroutine, rhs: Coroutine) -> Bool {
        lhs === rhs
    }
    
    @inlinable public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
