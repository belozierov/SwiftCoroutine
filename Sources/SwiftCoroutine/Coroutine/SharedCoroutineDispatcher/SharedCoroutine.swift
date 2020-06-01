//
//  SharedCoroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class SharedCoroutine {
    
    internal typealias CompletionState = SharedCoroutineQueue.CompletionState
    
    private struct StackBuffer {
        let stack: UnsafeMutableRawPointer, size: Int
    }
    
    internal let dispatcher: SharedCoroutineDispatcher
    internal let queue: SharedCoroutineQueue
    private(set) var scheduler: CoroutineScheduler
    
    private var state: Int = .running
    private var environment: UnsafeMutablePointer<CoroutineContext.SuspendData>!
    private var stackBuffer: StackBuffer!
    private var isCanceled = 0
    private var awaitTag = 0
    
    internal init(dispatcher: SharedCoroutineDispatcher, queue: SharedCoroutineQueue, scheduler: CoroutineScheduler) {
        self.dispatcher = dispatcher
        self.queue = queue
        self.scheduler = scheduler
    }
  
    // MARK: - Actions
    
    internal func start() -> CompletionState {
        performAsCurrent { perform(queue.context.start) }
    }
    
    internal func resume() -> CompletionState {
        performAsCurrent(resumeContext)
    }
    
    private func resumeContext() -> CompletionState {
        perform { queue.context.resume(from: environment.pointee.env) }
    }
    
    private func perform(_ block: () -> Bool) -> CompletionState {
        if block() { return .finished }
        while true {
            switch state {
            case .suspending:
                if atomicCAS(&state, expected: .suspending, desired: .suspended) {
                    return .suspended
                }
            case .running:
                return resumeContext()
            case .restarting:
                return .restarting
            default:
                return .suspended
            }
        }
    }
    
    private func suspend() {
        if environment == nil {
            environment = .allocate(capacity: 1)
            environment.initialize(to: .init())
        }
        queue.context.suspend(to: environment)
    }
    
    // MARK: - Stack
    
    internal func saveStack() {
        let size = environment.pointee.sp.distance(to: queue.context.stackTop)
        let stack = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 16)
        stack.copyMemory(from: environment.pointee.sp, byteCount: size)
        stackBuffer = .init(stack: stack, size: size)
    }
    
    internal func restoreStack() {
        environment.pointee.sp.copyMemory(from: stackBuffer.stack, byteCount: stackBuffer.size)
        stackBuffer.stack.deallocate()
        stackBuffer = nil
    }
    
    deinit {
        environment?.pointee.env.deallocate()
        environment?.deallocate()
    }
    
}

extension SharedCoroutine: CoroutineProtocol {
    
    internal func await<T>(_ callback: (@escaping (T) -> Void) -> Void) throws -> T {
        if isCanceled == 1 { throw CoroutineError.canceled }
        state = .suspending
        let tag = awaitTag
        var result: T!
        callback { value in
            while true {
                guard self.awaitTag == tag else { return }
                if atomicCAS(&self.awaitTag, expected: tag, desired: tag + 1) { break }
            }
            result = value
            self.resumeIfSuspended()
        }
        if state == .suspending { suspend() }
        if isCanceled == 1 { throw CoroutineError.canceled }
        return result
    }
    
    internal func await<T>(on scheduler: CoroutineScheduler, task: () throws -> T) throws -> T {
        if isCanceled == 1 { throw CoroutineError.canceled }
        let currentScheduler = self.scheduler
        setScheduler(scheduler)
        defer { setScheduler(currentScheduler) }
        if isCanceled == 1 { throw CoroutineError.canceled }
        return try task()

    }
    
    private func setScheduler(_ scheduler: CoroutineScheduler) {
        self.scheduler = scheduler
        state = .restarting
        suspend()
    }
    
    internal func cancel() {
        atomicStore(&isCanceled, value: 1)
        resumeIfSuspended()
    }
    
    private func resumeIfSuspended() {
        while true {
            switch state {
            case .suspending:
                if atomicCAS(&state, expected: .suspending, desired: .running) { return }
            case .suspended:
                if atomicCAS(&state, expected: .suspended, desired: .running) {
                    return queue.resume(coroutine: self)
                }
            default:
                return
            }
        }
    }
    
}

fileprivate extension Int {
    
    static let running = 0
    static let suspending = 1
    static let suspended = 2
    static let restarting = 3
    
}
