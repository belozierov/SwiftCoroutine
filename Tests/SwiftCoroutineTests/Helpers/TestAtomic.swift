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
    
    func testUpdate() {
        measure {
            let group = DispatchGroup()
            for _ in 0..<100_000 { group.enter() }
            var atomic = AtomicInt(wrappedValue: 0)
            XCTAssertEqual(atomic.wrappedValue, 0)
            DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
                while true {
                    let value = atomic.wrappedValue
                    if atomic.update(from: value, to: value + 1) { break }
                }
                group.leave()
            }
            group.wait()
            XCTAssertEqual(atomic.wrappedValue, 100_000)
        }
    }
    
}
