//
//  TestAtomic.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
import Dispatch
@testable import SwiftCoroutine

class TestAtomic: XCTestCase {
    
    func testTuple() {
        var tuple = AtomicTuple()
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            tuple.update { ($0.0 + 1, $0.1) }
            tuple.update { ($0.0, $0.1 + 1) }
        }
        XCTAssertEqual(tuple.value.0, 100_000)
        XCTAssertEqual(tuple.value.1, 100_000)
    }
    
    func testTuple2() {
        var atomic = AtomicTuple(), counter = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            while true {
                if atomic.update(keyPath: \.0, with: 1) == 0 {
                    counter += 1
                    atomic.value.0 = 0
                    break
                }
            }
        }
        XCTAssertEqual(counter, 100_000)
    }
    
    func testIntUpdate() {
        var atomic = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomicUpdate(&atomic) { $0 + 1 }
        }
        XCTAssertEqual(atomic, 100_000)
    }
    
    func testIntIncrease() {
        var atomic = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomicAdd(&atomic, value: 1)
        }
        XCTAssertEqual(atomic, 100_000)
    }
    
}
