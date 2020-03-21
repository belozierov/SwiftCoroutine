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
    private(set) var environment: UnsafeMutablePointer<CoroutineContext.SuspendData>!
    private var stackBuffer: StackBuffer!
    
    init(dispatcher: SharedCoroutineDispatcher, queue: SharedCoroutineQueue, scheduler: TaskScheduler) {
        self.dispatcher = dispatcher
        self.queue = queue
        self.scheduler = scheduler
    }
    
//    private func perform(_ block: () -> Bool) {
//        performAsCurrent { _perform(block) }
//    }
    
    private func perform(_ block: () -> Bool) {
        let isFinished = block()
        if $state.update({
            if $0 == .restarting { return .suspended }
            return isFinished ? .finished : .suspended
        }).old == .restarting {
            perform { queue.resume(self) }
        }
    }
    
    // MARK: - start
    
    func _start(_ task: @escaping () -> Void) {
        state = .running
        performAsCurrent { perform { queue.start(task) } }
    }
    
    // MARK: - resume
    
    func resume() throws {
        if $state.update({
            switch $0 {
            case .suspended: return .running
            case .suspending: return .restarting
            default: return $0
            }
        }).old == .suspended {
            dispatcher.resume(self)
        }
    }
    
    func _resume() {
        performAsCurrent { perform { queue.resume(self) } }
    }
    
    // MARK: - suspend
    
    func suspend() throws {
        guard $state.update(from: .running, to: .suspending)
            else { throw CoroutineError.wrongState }
        _suspend()
    }
    
    private func _suspend() {
        if environment == nil {
            environment = .allocate(capacity: 1)
            environment.initialize(to: .init())
        }
        queue.suspend(self)
    }
    
    // MARK: - await
    
    func await<T>(_ callback: (@escaping (T) -> Void) -> Void) throws -> T {
        var result: T?
        callback {
            result = $0
            if self.$state.update({ $0 == .suspended ? .running : .restarting }).old == .suspended {
                self.dispatcher.resume(self)
            }
        }
        if $state.update({ $0 == .running ? .suspending : .running }).new == .suspending {
            _suspend()
        }
        if let result = result { return result }
        throw CoroutineError.wrongState
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
