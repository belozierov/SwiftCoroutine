//
//  SharedCoroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif

final class SharedCoroutine: CoroutineProtocol {
    
    private enum State: Int {
        case running, suspending, suspended
    }
    
    private struct StackBuffer {
        let stack: UnsafeMutableRawPointer, size: Int
    }
    
    private let dispatcher: SharedCoroutineDispatcher
    let queue: SharedCoroutineQueue
    let scheduler: TaskScheduler
    private(set) var environment: UnsafeMutablePointer<CoroutineContext.SuspendData>!
    private var stackBuffer: StackBuffer!
    private var state = AtomicEnum(wrappedValue: State.running)
    
    init(dispatcher: SharedCoroutineDispatcher, queue: SharedCoroutineQueue, scheduler: TaskScheduler) {
        self.dispatcher = dispatcher
        self.queue = queue
        self.scheduler = scheduler
    }
    
    // MARK: - actions
    
    func start(_ task: @escaping () -> Void) {
        performAsCurrent { perform { queue.start(task) } }
    }
    
    func resume() {
        performAsCurrent { perform { queue.resume(self) } }
    }
    
    private func suspend() {
        if environment == nil {
            environment = .allocate(capacity: 1)
            environment.initialize(to: .init())
        }
        queue.suspend(self)
    }
    
    private func perform(_ block: () -> Bool) {
        if block() { return }
        if state.update(.suspended) == .running {
            state.value = .running
            perform { queue.resume(self) }
        }
    }
    
    // MARK: - await
    
    func await<T>(_ callback: (@escaping (T) -> Void) -> Void) -> T {
        state.value = .suspending
        var result: T!
        callback {
            result = $0
            if self.state.update(.running) == .suspended {
                self.dispatcher.resume(self)
            }
        }
        if state.value == .suspending { suspend() }
        return result
    }
    
    deinit {
        environment?.pointee.env.deallocate()
        environment?.deallocate()
    }
    
}

extension SharedCoroutine {
    
    func saveStack() {
        let size = environment.pointee.sp.distance(to: queue.context.stackTop)
        let stack = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 16)
        stack.copyMemory(from: environment.pointee.sp, byteCount: size)
        stackBuffer = .init(stack: stack, size: size)
    }
    
    func restoreStack() {
        environment.pointee.sp.copyMemory(from: stackBuffer.stack, byteCount: stackBuffer.size)
        stackBuffer.stack.deallocate()
        stackBuffer = nil
    }
    
}
