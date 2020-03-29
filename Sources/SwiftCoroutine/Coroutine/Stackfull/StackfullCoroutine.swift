////
////  StackfullCoroutine.swift
////  SwiftCoroutine
////
////  Created by Alex Belozierov on 07.03.2020.
////  Copyright © 2020 Alex Belozierov. All rights reserved.
////
//
//final class StackfullCoroutine: CoroutineProtocol {
//
//    private enum State: Int {
//        case running, suspending, suspended
//    }
//
//    let context: CoroutineContext
//    let scheduler: CoroutineScheduler
//    @AtomicIntRepresentable private(set) var state: State = .prepared
//    private var resumeEnv: UnsafeMutablePointer<Int32>!
//
//    init(context: CoroutineContext, scheduler: TaskScheduler, completion: (() -> Void)? = nil) {
//        self.context = context
//        self.scheduler = scheduler
//        self.completion = completion.map { completion in { if $0 { completion() } } }
//    }
//
//    func resume() throws {
//        while true {
//            switch state {
//            case .prepared where $state.update(from: .prepared, to: .running):
//                return perform(block: context.start)
//            case .suspended where $state.update(from: .suspended, to: .running):
//                return perform { self.context.resume(from: self.resumeEnv) }
//            default:
//                throw CoroutineError.wrongState
//            }
//        }
//    }
//
//    private func suspend() {
//        if resumeEnv == nil { resumeEnv = .allocate(capacity: .environmentSize) }
//        context.suspend(to: resumeEnv)
//    }
//
//    private func perform(block: @escaping () -> Bool) {
//        func execute() {
//            performAsCurrent {
//                let isFinished = block()
//                state = isFinished ? .finished : .suspended
//                completion?(isFinished)
//            }
//        }
//        scheduler.isCurrent() ? execute() : scheduler.execute(execute)
//    }
//
//    deinit {
//        resumeEnv?.deallocate()
//    }
//
//}
