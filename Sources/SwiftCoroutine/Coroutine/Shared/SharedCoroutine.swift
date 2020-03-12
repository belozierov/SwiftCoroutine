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
    
    private struct StackBuffer {
        let stack: UnsafeMutableRawPointer, size: Int
    }
    
    private let dispatcher: SharedCoroutineDispatcher
    let queue: SharedCoroutineQueue
    let scheduler: TaskScheduler
    
    @AtomicIntRepresentable private(set) var state: State = .prepared
    private(set) var environment: UnsafeMutablePointer<__CoroutineEnvironment>!
    private var stackBuffer: StackBuffer!
    private var completion: (() -> Void)?
    
    init(dispatcher: SharedCoroutineDispatcher, queue: SharedCoroutineQueue, scheduler: TaskScheduler) {
        self.dispatcher = dispatcher
        self.queue = queue
        self.scheduler = scheduler
    }
    
    private func perform(_ block: () -> Bool) {
        performAsCurrent {
            let isFinished = block()
            state = isFinished ? .finished : .suspended
            completion?()
            completion = nil
        }
    }
    
    // MARK: - start
    
    func start() {}
    
    func _start(_ task: @escaping () -> Void) {
        state = .running
        perform { queue.start(task) }
    }
    
    // MARK: - resume
    
    func resume() {
        state = .running
        dispatcher.resume(self)
    }
    
    func _resume() {
        perform { queue.resume(self) }
    }
    
    // MARK: - suspend
    
    func suspend() {
        state = .suspending
        _suspend()
    }
    
    func suspend(with completion: @escaping () -> Void) {
        state = .suspending
        self.completion = completion
        _suspend()
    }
    
    private func _suspend() {
        if environment == nil {
            environment = .allocate(capacity: 1)
            environment.initialize(to: .init())
        }
        queue.suspend(self)
    }
    
    deinit {
        environment?.pointee.env?.deallocate()
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
