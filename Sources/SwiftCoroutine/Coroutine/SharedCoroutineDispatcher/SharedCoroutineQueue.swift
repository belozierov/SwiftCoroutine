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
    private var coroutine: SharedCoroutine?
    
    private(set) var started = 0
    private var atomic = AtomicTuple()
    private var prepared = BlockingFifoQueue<SharedCoroutine>()
    
    internal init(stackSize size: Int) {
        context = CoroutineContext(stackSize: size)
    }
    
    internal func occupy() -> Bool {
        atomic.update(keyPath: \.0, with: .running) == .isFree
    }
    
    // MARK: - Actions
    
    internal func start(dispatcher: SharedCoroutineDispatcher, scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        let coroutine: SharedCoroutine
        if let previous = self.coroutine, previous.dispatcher == nil {
            coroutine = previous
            coroutine.scheduler = scheduler
            coroutine.dispatcher = dispatcher
        } else {
            self.coroutine?.saveStack()
            coroutine = SharedCoroutine(dispatcher: dispatcher, queue: self,
                                        scheduler: scheduler)
            self.coroutine = coroutine
        }
        started += 1
        context.block = task
        complete(coroutine: coroutine, state: coroutine.start())
    }
    
    internal func resume(coroutine: SharedCoroutine) {
        let wasFree = atomic.update { state, count in
            (.running, state == .isFree ? count : count + 1)
        }.old.0 == .isFree
        wasFree ? resumeOnQueue(coroutine) : prepared.push(coroutine)
    }
    
    private func resumeOnQueue(_ coroutine: SharedCoroutine) {
        if self.coroutine !== coroutine {
            if let coroutine = self.coroutine, coroutine.dispatcher != nil {
                coroutine.saveStack()
            }
            coroutine.restoreStack()
            self.coroutine = coroutine
        }
        coroutine.scheduler.scheduleTask {
            self.complete(coroutine: coroutine, state: coroutine.resume())
        }
    }
    
    private func complete(coroutine: SharedCoroutine, state: CompletionState) {
        switch state {
        case .finished:
            started -= 1
            let dispatcher = coroutine.dispatcher!
            coroutine.reset()
            performNext(for: dispatcher)
        case .suspended:
            performNext(for: coroutine.dispatcher)
        case .restarting:
            coroutine.scheduler.scheduleTask {
                self.complete(coroutine: coroutine, state: coroutine.resume())
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
