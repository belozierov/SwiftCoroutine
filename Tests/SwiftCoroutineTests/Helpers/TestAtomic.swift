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
    
    enum State: Int {
        case free, blocked
    }
    
    func testEnumUpdate() {
        var atomic = AtomicEnum(value: State.free), counter = 0
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            while true {
                if atomic.update(.blocked) == .free {
                    counter += 1
                    atomic.value = .free
                    break
                }
            }
        }
        XCTAssertEqual(counter, 100_000)
    }
    
    func testIntUpdate() {
        var atomic = AtomicInt(value: 0)
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomic.update { $0 + 1 }
        }
        XCTAssertEqual(atomic.value, 100_000)
    }
    
    func testIntIncrease() {
        var atomic = AtomicInt(value: 0)
        DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
            atomic.add(1)
        }
        XCTAssertEqual(atomic.value, 100_000)
    }
    
}
