//
//  PseudoCoroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 05.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline internal final class PseudoCoroutine: CoroutineProtocol {
    
    @usableFromInline internal static let shared = PseudoCoroutine()
    
    @usableFromInline
    internal func await<T>(_ callback: (@escaping (T) -> Void) -> Void) -> T {
        let condition = PsxCondition()
        defer { condition.free() }
        var result: T!
        callback {
            if result != nil { return }
            condition.lock()
            result = $0
            condition.signal()
            condition.unlock()
        }
        condition.lock()
        while result == nil { condition.wait() }
        condition.unlock()
        return result
    }
    
    @usableFromInline
    internal func await<T>(on scheduler: CoroutineScheduler, task: () throws -> T) rethrows -> T {
        try withoutActuallyEscaping(task) {
            try withUnsafePointer(to: $0) { task in
                try self.await { callback in
                    scheduler.scheduleTask {
                        callback(Result(catching: task.pointee))
                    }
                }.get()
            }
        }
    }
    
}
