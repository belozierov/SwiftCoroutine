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
    
    func test() {
        var mask = AtomicBitMask()
        mask.insert(3)
        mask.insert(60)
        mask.insert(5)
        XCTAssertEqual(mask.pop(offset: 4), 60)
        XCTAssertEqual(mask.pop(offset: 4), 3)
        XCTAssertEqual(mask.pop(offset: 4), 5)
    }
    
    func testThreadSafeFifoQueues() {
        var queue = ThreadSafeFifoQueues<Int>()
        DispatchQueue.concurrentPerform(iterations: 100_000) { index in
            queue.push(index)
            XCTAssertNotNil(queue.pop())
        }
        queue.free()
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
