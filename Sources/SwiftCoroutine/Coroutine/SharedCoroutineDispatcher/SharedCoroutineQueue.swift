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
    
    internal let tag: Int
    internal let context: CoroutineContext
    private let storage: SharedCoroutineStorage
    
    private var prepared = ThreadSaveFifoQueue<SharedCoroutine>()
    private var coroutine: SharedCoroutine?
    private(set) var started = 0
    
    internal init(storage: SharedCoroutineStorage, tag: Int, stackSize size: Int) {
        self.storage = storage
        self.tag = tag
        context = CoroutineContext(stackSize: size)
    }
    
    var hasPrepared: Bool {
        !prepared.isEmpty
    }
    
    func startPrepared() {
        prepared.pop().map(resumeOnQueue)
    }
    
    // MARK: - Actions
    
    internal func start(dispatcher: SharedCoroutineDispatcher,
                        scheduler: CoroutineScheduler, task: @escaping () -> Void) {
        let coroutine: SharedCoroutine
        if let previous = self.coroutine, previous.dispatcher == nil {
            coroutine = previous
            coroutine.dispatcher = dispatcher
            coroutine.scheduler = scheduler
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
        storage.removeSuspended(with: tag)
            ? resumeOnQueue(coroutine)
            : prepared.push(coroutine)
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
            self.coroutine?.reset()
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
        if let coroutine = prepared.pop() {
            resumeOnQueue(coroutine)
        } else {
            dispatcher.receiveQueue(self)
        }
    }
    
    deinit {
        prepared.free()
    }
    
}

