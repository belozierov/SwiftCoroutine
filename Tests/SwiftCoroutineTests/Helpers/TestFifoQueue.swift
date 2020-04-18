//
//  TestFifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class TestFifoQueue: XCTestCase {
    
    func testThreadSafeFifoQueues() {
        var queue = ThreadSafeFifoQueues<Int>()
        DispatchQueue.concurrentPerform(iterations: 100_000) { index in
            queue.push(index)
            XCTAssertNotNil(queue.pop())
        }
        queue.free()
    }
    
    func testThreadSafeFifoQueues2() {
        var queue = ThreadSafeFifoQueues<Int>()
        queue.push(0)
        queue.push(3)
        XCTAssertEqual(queue.pop(), 0)
        queue.insertAtStart(2)
        queue.insertAtStart(1)
        queue.push(4)
        queue.push(5)
        queue.insertAtStart(0)
        for i in 0..<6 {
            XCTAssertEqual(queue.pop(), i)
        }
    }
    
    func testQueue() {
        var queue = FifoQueue<Int>()
        for i in 0..<100 { queue.push(i) }
        for i in 0..<50 { XCTAssertEqual(queue.pop(), i) }
        for i in 100..<200 { queue.push(i) }
        for i in 50..<200 { XCTAssertEqual(queue.pop(), i) }
        XCTAssertNil(queue.pop())
    }
    
}
