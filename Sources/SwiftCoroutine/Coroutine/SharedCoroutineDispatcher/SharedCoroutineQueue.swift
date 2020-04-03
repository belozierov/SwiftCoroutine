//
//  SharedCoroutineQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class SharedCoroutineQueue {
    
    internal typealias Task = SharedCoroutineDispatcher.Task
    
    internal enum CompletionState {
        case finished, suspended, restarting
    }
    
    internal let context: CoroutineContext
    private var coroutine: SharedCoroutine?
    private var prepared = FifoQueue<SharedCoroutine>()
    private(set) var started = 0
    
    internal init(stackSize size: Int) {
        context = CoroutineContext(stackSize: size)
    }
    
    // MARK: - Queue
    
    internal func push(_ coroutine: SharedCoroutine) {
        prepared.push(coroutine)
    }
    
    internal func pop() -> SharedCoroutine? {
        prepared.pop()
    }
    
    // MARK: - Actions
    
    internal func start(dispatcher: SharedCoroutineDispatcher, task: Task) {
        coroutine?.saveStack()
        let coroutine = SharedCoroutine(dispatcher: dispatcher, queue: self,
                                        scheduler: task.scheduler)
        self.coroutine = coroutine
        started += 1
        context.block = task.task
        complete(coroutine: coroutine, state: coroutine.start())
    }
    
    internal func resume(coroutine: SharedCoroutine) {
        if self.coroutine !== coroutine {
            self.coroutine?.saveStack()
            coroutine.restoreStack()
            self.coroutine = coroutine
        }
        complete(coroutine: coroutine, state: coroutine.resume())
    }
    
    private func complete(coroutine: SharedCoroutine, state: CompletionState) {
        switch state {
        case .finished:
            started -= 1
            self.coroutine = nil
            coroutine.dispatcher.performNext(for: self)
        case .suspended:
            coroutine.dispatcher.performNext(for: self)
        case .restarting:
            coroutine.scheduler.scheduleTask {
                self.complete(coroutine: coroutine, state: coroutine.resume())
            }
        }
    }
    
}

extension SharedCoroutineQueue: Hashable {
    
    @inlinable internal static func == (lhs: SharedCoroutineQueue, rhs: SharedCoroutineQueue) -> Bool {
        lhs === rhs
    }
    
    @inlinable internal func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
