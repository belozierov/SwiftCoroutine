//
//  SharedCoroutineQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal final class SharedCoroutineQueue {
    
    private struct Task {
        let scheduler: CoroutineScheduler, task: () -> Void
    }
    
    internal enum CompletionState {
        case finished, suspended, restarting
    }
    
    internal let context: CoroutineContext
    private var coroutine: SharedCoroutine
    
    private(set) var started = 0
    private var atomic = AtomicTuple()
    private var prepared = BlockingFifoQueue<SharedCoroutine>()
    
    internal init(stackSize size: Int) {
        context = CoroutineContext(stackSize: size)
        coroutine = SharedCoroutine()
    }
    
    internal func occupy() -> Bool {
        atomic.update(keyPath: \.0, with: .running) == .isFree
    }
    
    // MARK: - Actions
    
    internal func start(dispatcher: SharedCoroutineDispatcher, scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        if coroutine.configuration != nil {
            coroutine.saveStack()
            coroutine = SharedCoroutine()
        }
        coroutine.configuration = .init(dispatcher: dispatcher, queue: self, scheduler: scheduler)
        started += 1
        context.block = task
        complete(with: coroutine.start())
    }
    
    internal func resume(coroutine: SharedCoroutine) {
        let wasFree = atomic.update { state, count in
            (.running, state == .isFree ? count : count + 1)
        }.old.0 == .isFree
        wasFree ? resumeOnQueue(coroutine) : prepared.push(coroutine)
    }
    
    private func resumeOnQueue(_ coroutine: SharedCoroutine) {
        if self.coroutine !== coroutine {
            if self.coroutine.configuration != nil {
                self.coroutine.saveStack()
            }
            coroutine.restoreStack()
            self.coroutine = coroutine
        }
        coroutine.configuration.scheduler.scheduleTask {
            self.complete(with: coroutine.resume())
        }
    }
    
    private func complete(with state: CompletionState) {
        switch state {
        case .finished:
            started -= 1
            let dispatcher = coroutine.configuration.dispatcher
            coroutine.reset()
            performNext(for: dispatcher)
        case .suspended:
            performNext(for: coroutine.configuration.dispatcher)
        case .restarting:
            coroutine.configuration.scheduler.scheduleTask {
                self.complete(with: self.coroutine.resume())
            }
        }
    }
    
    private func performNext(for dispatcher: SharedCoroutineDispatcher) {
        let isFinished = atomic.update { _, count in
            count > 0 ? (.running, count - 1) : (.isFree, 0)
        }.new.0 == .isFree
        isFinished ? dispatcher.push(self) : resumeOnQueue(prepared.pop())
    }
    
    deinit {
        prepared.free()
    }
    
}

extension Int32 {
    
    fileprivate static let running: Int32 = 0
    fileprivate static let isFree: Int32 = 1
    
}
