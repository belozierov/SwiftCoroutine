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
    
    var inQueue = false
    private(set) var started = 0
    private var atomic = AtomicTuple()
    private var prepared = FifoQueue<SharedCoroutine>()
    
    internal init(stackSize size: Int) {
        context = CoroutineContext(stackSize: size)
        coroutine = SharedCoroutine()
    }
    
    internal func occupy() -> Bool {
        atomic.update(keyPath: \.0, with: .running) == .isFree
    }
    
    // MARK: - Actions
    
    internal func start(dispatcher: SharedCoroutineDispatcher, scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        if coroutine.dispatcher != nil {
            coroutine.saveStack()
            coroutine = SharedCoroutine()
        }
        coroutine.dispatcher = dispatcher
        coroutine.queue = self
        coroutine.scheduler = scheduler
        started += 1
        context.block = task
        complete(with: coroutine.start())
    }
    
    internal func resume(coroutine: SharedCoroutine) {
        let wasFree = atomic.update { state, count in
            if state == .isFree {
                return (.running, count)
            } else {
                return (.running, count + 1)
            }
        }.old.0 == .isFree
        wasFree ? resumeOnQueue(coroutine) : prepared.push(coroutine)
    }
    
    private func resumeOnQueue(_ coroutine: SharedCoroutine) {
        if self.coroutine !== coroutine {
            if self.coroutine.dispatcher != nil {
                self.coroutine.saveStack()
            }
            coroutine.restoreStack()
            self.coroutine = coroutine
        }
        coroutine.scheduler.scheduleTask {
            self.complete(with: coroutine.resume())
        }
    }
    
    private func complete(with state: CompletionState) {
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
                self.complete(with: self.coroutine.resume())
            }
        }
    }
    
    private func performNext(for dispatcher: SharedCoroutineDispatcher) {
        let isFinished = atomic.update { _, count in
            count > 0 ? (.running, count - 1) : (.isFree, 0)
        }.new.0 == .isFree
        isFinished ? dispatcher.push(self) : resumeOnQueue(prepared.blockingPop())
    }
    
    deinit {
        prepared.free()
    }
    
}

extension Int32 {
    
    fileprivate static let running: Int32 = 0
    fileprivate static let isFree: Int32 = 1
    
}
