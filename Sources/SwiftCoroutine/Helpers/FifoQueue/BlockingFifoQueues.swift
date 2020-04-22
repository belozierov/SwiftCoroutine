//
//  BlockingFifoQueues.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct BlockingFifoQueues<T> {
    
    private let number: Int
    private let queues: UnsafeMutablePointer<BlockingFifoQueue<T>>
    private var pushIndex = AtomicInt()
    private var popIndex = AtomicInt()
    
    internal init(number: Int = .processorsNumber) {
        self.number = number
        queues = .allocate(capacity: number)
        (0..<number).forEach { (queues + $0).initialize(to: BlockingFifoQueue()) }
    }
    
    internal mutating func push(_ item: T) {
        let index = pushIndex.update { $0 + 1 < number ? $0 + 1 : 0 }.old
        (queues + index).pointee.push(item)
    }
    
    internal mutating func pop() -> T {
        let index = popIndex.update { $0 + 1 < number ? $0 + 1 : 0 }.old
        return (queues + index).pointee.pop()
    }
    
    internal func forEach(_ body: (T) -> Void) {
        (0..<number).forEach { (queues + $0).pointee.forEach(body) }
    }
    
    internal func free() {
        (0..<number).forEach { queues[$0].free() }
        queues.deinitialize(count: number)
        queues.deallocate()
    }
    
}
