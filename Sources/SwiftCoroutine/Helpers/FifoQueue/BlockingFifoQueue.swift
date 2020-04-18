//
//  BlockingFifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 17.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct BlockingFifoQueue<T> {
    
    private let condition = PsxCondition()
    private var queue = FifoQueue<T>()
    private var waiting = 0
    
    internal mutating func insertAtStart(_ item: T) {
        condition.lock()
        queue.insertAtStart(item)
        if waiting != 0 { condition.signal() }
        condition.unlock()
    }
    
    internal mutating func push(_ item: T) {
        condition.lock()
        queue.push(item)
        if waiting != 0 { condition.signal() }
        condition.unlock()
    }
    
    internal mutating func pop() -> T {
        condition.lock()
        repeat {
            if let item = queue.pop() {
                condition.unlock()
                return item
            }
            waiting += 1
            condition.wait()
            waiting -= 1
        } while true
    }
    
    internal func free() {
        condition.free()
    }
    
}
